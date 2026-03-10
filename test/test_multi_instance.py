"""Tests for multi-instance provider support (#117)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure lib is on path
_lib = Path(__file__).resolve().parent.parent / "lib"
sys.path.insert(0, str(_lib))

# ── Stub out modules that use Python 3.10+ syntax (str | None) ─────────────
# ccb_config and terminal use union-type syntax unsupported on Python <3.10.
# We pre-populate sys.modules with lightweight stubs so that caskd_session /
# gaskd_session can be imported without hitting a TypeError.

def _ensure_stub(mod_name: str, attrs: dict[str, Any] | None = None) -> None:
    """Insert a stub module into sys.modules if not already importable."""
    if mod_name in sys.modules:
        return
    try:
        __import__(mod_name)
        return
    except Exception:
        pass
    stub = types.ModuleType(mod_name)
    for k, v in (attrs or {}).items():
        setattr(stub, k, v)
    sys.modules[mod_name] = stub


_ensure_stub("terminal", {
    "_subprocess_kwargs": lambda: {},
    "get_backend_for_session": lambda data: None,
})
_ensure_stub("ccb_config", {
    "apply_backend_env": lambda: None,
    "get_backend_env": lambda: None,
})
_ensure_stub("project_id", {
    "compute_ccb_project_id": lambda p: "stub_project_id",
})


# ── parse_qualified_provider ────────────────────────────────────────────────

class TestParseQualifiedProvider:
    def test_plain_provider(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider("codex") == ("codex", None)

    def test_qualified_provider(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider("codex:auth") == ("codex", "auth")

    def test_empty_string(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider("") == ("", None)

    def test_none_input(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider(None) == ("", None)

    def test_colon_only(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider(":") == ("", None)

    def test_provider_with_empty_instance(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider("codex:") == ("codex", None)

    def test_uppercase_normalized(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider("CODEX:Auth") == ("codex", "auth")

    def test_whitespace_trimmed(self):
        from providers import parse_qualified_provider
        assert parse_qualified_provider(" codex : auth ") == ("codex", "auth")

    def test_multiple_colons(self):
        """Only split on first colon."""
        from providers import parse_qualified_provider
        base, instance = parse_qualified_provider("codex:auth:extra")
        assert base == "codex"
        assert instance == "auth:extra"

    def test_all_providers(self):
        from providers import parse_qualified_provider
        for prov in ("codex", "gemini", "opencode", "claude", "droid", "copilot", "codebuddy", "qwen"):
            assert parse_qualified_provider(prov) == (prov, None)
            assert parse_qualified_provider(f"{prov}:test") == (prov, "test")


# ── make_qualified_key ──────────────────────────────────────────────────────

class TestMakeQualifiedKey:
    def test_no_instance(self):
        from providers import make_qualified_key
        assert make_qualified_key("codex", None) == "codex"

    def test_with_instance(self):
        from providers import make_qualified_key
        assert make_qualified_key("codex", "auth") == "codex:auth"

    def test_empty_instance(self):
        from providers import make_qualified_key
        assert make_qualified_key("codex", "") == "codex"

    def test_roundtrip(self):
        from providers import parse_qualified_provider, make_qualified_key
        key = make_qualified_key("gemini", "frontend")
        base, inst = parse_qualified_provider(key)
        assert base == "gemini"
        assert inst == "frontend"

    def test_roundtrip_no_instance(self):
        from providers import parse_qualified_provider, make_qualified_key
        key = make_qualified_key("codex", None)
        base, inst = parse_qualified_provider(key)
        assert base == "codex"
        assert inst is None


# ── session_filename_for_instance ───────────────────────────────────────────

class TestSessionFilenameForInstance:
    def test_no_instance(self):
        from providers import session_filename_for_instance
        assert session_filename_for_instance(".codex-session", None) == ".codex-session"

    def test_with_instance(self):
        from providers import session_filename_for_instance
        assert session_filename_for_instance(".codex-session", "auth") == ".codex-auth-session"

    def test_empty_instance(self):
        from providers import session_filename_for_instance
        assert session_filename_for_instance(".codex-session", "") == ".codex-session"

    def test_whitespace_instance(self):
        from providers import session_filename_for_instance
        assert session_filename_for_instance(".codex-session", "  ") == ".codex-session"

    def test_all_provider_sessions(self):
        from providers import session_filename_for_instance
        cases = [
            (".codex-session", "auth", ".codex-auth-session"),
            (".gemini-session", "frontend", ".gemini-frontend-session"),
            (".opencode-session", "backend", ".opencode-backend-session"),
            (".claude-session", "review", ".claude-review-session"),
            (".droid-session", "test", ".droid-test-session"),
            (".copilot-session", "dev", ".copilot-dev-session"),
            (".codebuddy-session", "prod", ".codebuddy-prod-session"),
            (".qwen-session", "api", ".qwen-api-session"),
        ]
        for base, inst, expected in cases:
            assert session_filename_for_instance(base, inst) == expected, f"Failed for {base} + {inst}"


# ── Session module instance support ────────────────────────────────────────

class TestSessionModuleInstance:
    """Test that session modules accept instance parameter."""

    def test_codex_find_session_file_default(self, tmp_path):
        from caskd_session import find_project_session_file
        # No session file exists -- should return None
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_codex_find_session_file_with_instance(self, tmp_path):
        from caskd_session import find_project_session_file
        # Create instance-specific session file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".codex-auth-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="auth")
        assert result is not None
        assert "auth" in result.name

    def test_codex_load_session_no_instance(self, tmp_path):
        from caskd_session import load_project_session
        result = load_project_session(tmp_path)
        assert result is None

    def test_codex_load_session_with_instance(self, tmp_path):
        from caskd_session import load_project_session
        result = load_project_session(tmp_path, instance="auth")
        assert result is None  # No file exists

    def test_codex_load_session_with_instance_file_exists(self, tmp_path):
        from caskd_session import load_project_session
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".codex-auth-session"
        session_file.write_text('{"pane_id": "%42", "work_dir": "/tmp/test"}')
        result = load_project_session(tmp_path, instance="auth")
        assert result is not None
        assert result.pane_id == "%42"

    def test_codex_compute_session_key_no_instance(self):
        from caskd_session import compute_session_key, CodexProjectSession
        session = CodexProjectSession(
            session_file=Path("/tmp/test/.ccb/.codex-session"),
            data={"ccb_project_id": "abc123", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session)
        assert key == "codex:abc123"

    def test_codex_compute_session_key_with_instance(self):
        from caskd_session import compute_session_key, CodexProjectSession
        session = CodexProjectSession(
            session_file=Path("/tmp/test/.ccb/.codex-auth-session"),
            data={"ccb_project_id": "abc123", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="auth")
        assert "auth" in key
        assert "abc123" in key

    def test_gemini_find_session_file_default(self, tmp_path):
        from gaskd_session import find_project_session_file
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_gemini_find_session_file_with_instance(self, tmp_path):
        from gaskd_session import find_project_session_file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".gemini-frontend-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="frontend")
        assert result is not None
        assert "frontend" in result.name

    def test_gemini_compute_session_key_with_instance(self):
        from gaskd_session import compute_session_key, GeminiProjectSession
        session = GeminiProjectSession(
            session_file=Path("/tmp/test/.ccb/.gemini-session"),
            data={"ccb_project_id": "xyz789", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="frontend")
        assert "frontend" in key
        assert "xyz789" in key


# ── ProviderRequest instance field ──────────────────────────────────────────

class TestProviderRequestInstance:
    def test_default_instance_is_none(self):
        from askd.adapters.base import ProviderRequest
        req = ProviderRequest(
            client_id="test",
            work_dir="/tmp",
            timeout_s=30.0,
            quiet=False,
            message="hello",
            caller="manual",
        )
        assert req.instance is None

    def test_instance_can_be_set(self):
        from askd.adapters.base import ProviderRequest
        req = ProviderRequest(
            client_id="test",
            work_dir="/tmp",
            timeout_s=30.0,
            quiet=False,
            message="hello",
            caller="manual",
            instance="auth",
        )
        assert req.instance == "auth"


# ── Backward compatibility ──────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Verify that no-instance usage produces identical behavior."""

    def test_parse_qualified_plain(self):
        from providers import parse_qualified_provider
        for prov in ("codex", "gemini", "opencode", "claude", "droid", "copilot", "codebuddy", "qwen"):
            base, inst = parse_qualified_provider(prov)
            assert base == prov
            assert inst is None

    def test_session_filename_unchanged(self):
        from providers import session_filename_for_instance
        for name in (".codex-session", ".gemini-session", ".opencode-session",
                     ".claude-session", ".droid-session", ".copilot-session",
                     ".codebuddy-session", ".qwen-session"):
            assert session_filename_for_instance(name, None) == name

    def test_make_qualified_key_no_instance(self):
        from providers import make_qualified_key
        assert make_qualified_key("codex", None) == "codex"


# ── Daemon routing with instance ────────────────────────────────────────────

class TestDaemonInstanceRouting:
    """Test that the daemon correctly parses and routes instance-qualified providers."""

    def test_daemon_parses_qualified_provider(self):
        """Verify _handle_request parses 'codex:auth' correctly."""
        from providers import parse_qualified_provider
        base, inst = parse_qualified_provider("codex:auth")
        assert base == "codex"
        assert inst == "auth"

    def test_pool_key_includes_instance(self):
        """Pool key should use qualified key for instance isolation."""
        from providers import make_qualified_key
        key = make_qualified_key("codex", "auth")
        assert key == "codex:auth"
        # Different instance = different pool key
        key2 = make_qualified_key("codex", "payment")
        assert key2 == "codex:payment"
        assert key != key2

    def test_same_provider_different_instances_isolated(self):
        """Two instances of the same provider must produce different keys."""
        from providers import make_qualified_key
        keys = set()
        for inst in ("auth", "payment", "frontend", "backend"):
            keys.add(make_qualified_key("codex", inst))
        assert len(keys) == 4

    def test_different_providers_same_instance_isolated(self):
        """Same instance name on different providers must produce different keys."""
        from providers import make_qualified_key
        key1 = make_qualified_key("codex", "auth")
        key2 = make_qualified_key("gemini", "auth")
        assert key1 != key2


# ── Remaining session module instance support ────────────────────────────

class TestRemainingSessionModules:
    """Test the remaining 6 session modules for instance support."""

    # ── opencode (oaskd_session) ──────────────────────────────────────────

    def test_opencode_find_session_file_default(self, tmp_path):
        from oaskd_session import find_project_session_file
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_opencode_find_session_file_with_instance(self, tmp_path):
        from oaskd_session import find_project_session_file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".opencode-test-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="test")
        assert result is not None
        assert "test" in result.name

    def test_opencode_load_session_no_file(self, tmp_path):
        from oaskd_session import load_project_session
        result = load_project_session(tmp_path, instance="test")
        assert result is None

    def test_opencode_compute_session_key_with_instance(self):
        from oaskd_session import compute_session_key, OpenCodeProjectSession
        session = OpenCodeProjectSession(
            session_file=Path("/tmp/test/.ccb/.opencode-session"),
            data={"ccb_project_id": "proj1", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="test")
        assert "test" in key
        assert "proj1" in key

    # ── claude (laskd_session) ────────────────────────────────────────────

    def test_claude_find_session_file_default(self, tmp_path):
        from laskd_session import find_project_session_file
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_claude_find_session_file_with_instance(self, tmp_path):
        from laskd_session import find_project_session_file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".claude-test-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="test")
        assert result is not None
        assert "test" in result.name

    def test_claude_load_session_no_file(self, tmp_path):
        from laskd_session import load_project_session
        result = load_project_session(tmp_path, instance="test")
        assert result is None

    def test_claude_compute_session_key_with_instance(self):
        from laskd_session import compute_session_key, ClaudeProjectSession
        session = ClaudeProjectSession(
            session_file=Path("/tmp/test/.ccb/.claude-session"),
            data={"ccb_project_id": "proj2", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="test")
        assert "test" in key
        assert "proj2" in key

    # ── droid (daskd_session) ─────────────────────────────────────────────

    def test_droid_find_session_file_default(self, tmp_path):
        from daskd_session import find_project_session_file
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_droid_find_session_file_with_instance(self, tmp_path):
        from daskd_session import find_project_session_file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".droid-test-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="test")
        assert result is not None
        assert "test" in result.name

    def test_droid_load_session_no_file(self, tmp_path):
        from daskd_session import load_project_session
        result = load_project_session(tmp_path, instance="test")
        assert result is None

    def test_droid_compute_session_key_with_instance(self):
        from daskd_session import compute_session_key, DroidProjectSession
        session = DroidProjectSession(
            session_file=Path("/tmp/test/.ccb/.droid-session"),
            data={"ccb_project_id": "proj3", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="test")
        assert "test" in key
        assert "proj3" in key

    # ── copilot (haskd_session) ───────────────────────────────────────────

    def test_copilot_find_session_file_default(self, tmp_path):
        from haskd_session import find_project_session_file
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_copilot_find_session_file_with_instance(self, tmp_path):
        from haskd_session import find_project_session_file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".copilot-test-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="test")
        assert result is not None
        assert "test" in result.name

    def test_copilot_load_session_no_file(self, tmp_path):
        from haskd_session import load_project_session
        result = load_project_session(tmp_path, instance="test")
        assert result is None

    def test_copilot_compute_session_key_with_instance(self):
        from haskd_session import compute_session_key, CopilotProjectSession
        session = CopilotProjectSession(
            session_file=Path("/tmp/test/.ccb/.copilot-session"),
            data={"ccb_project_id": "proj4", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="test")
        assert "test" in key
        assert "proj4" in key

    # ── codebuddy (baskd_session) ─────────────────────────────────────────

    def test_codebuddy_find_session_file_default(self, tmp_path):
        from baskd_session import find_project_session_file
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_codebuddy_find_session_file_with_instance(self, tmp_path):
        from baskd_session import find_project_session_file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".codebuddy-test-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="test")
        assert result is not None
        assert "test" in result.name

    def test_codebuddy_load_session_no_file(self, tmp_path):
        from baskd_session import load_project_session
        result = load_project_session(tmp_path, instance="test")
        assert result is None

    def test_codebuddy_compute_session_key_with_instance(self):
        from baskd_session import compute_session_key, CodebuddyProjectSession
        session = CodebuddyProjectSession(
            session_file=Path("/tmp/test/.ccb/.codebuddy-session"),
            data={"ccb_project_id": "proj5", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="test")
        assert "test" in key
        assert "proj5" in key

    # ── qwen (qaskd_session) ─────────────────────────────────────────────

    def test_qwen_find_session_file_default(self, tmp_path):
        from qaskd_session import find_project_session_file
        result = find_project_session_file(tmp_path)
        assert result is None

    def test_qwen_find_session_file_with_instance(self, tmp_path):
        from qaskd_session import find_project_session_file
        ccb_dir = tmp_path / ".ccb"
        ccb_dir.mkdir()
        session_file = ccb_dir / ".qwen-test-session"
        session_file.write_text('{"pane_id": "test"}')
        result = find_project_session_file(tmp_path, instance="test")
        assert result is not None
        assert "test" in result.name

    def test_qwen_load_session_no_file(self, tmp_path):
        from qaskd_session import load_project_session
        result = load_project_session(tmp_path, instance="test")
        assert result is None

    def test_qwen_compute_session_key_with_instance(self):
        from qaskd_session import compute_session_key, QwenProjectSession
        session = QwenProjectSession(
            session_file=Path("/tmp/test/.ccb/.qwen-session"),
            data={"ccb_project_id": "proj6", "work_dir": "/tmp/test"},
        )
        key = compute_session_key(session, instance="test")
        assert "test" in key
        assert "proj6" in key


# ── Adapter instance passing ─────────────────────────────────────────────

class TestAdapterInstancePassing:
    """Test that adapters accept instance parameter in load_session / compute_session_key."""

    def _assert_adapter_has_instance_params(self, adapter_cls):
        """Verify that load_session and compute_session_key accept instance."""
        import inspect
        load_sig = inspect.signature(adapter_cls.load_session)
        assert "instance" in load_sig.parameters, (
            f"{adapter_cls.__name__}.load_session missing 'instance' param"
        )
        key_sig = inspect.signature(adapter_cls.compute_session_key)
        assert "instance" in key_sig.parameters, (
            f"{adapter_cls.__name__}.compute_session_key missing 'instance' param"
        )

    def test_codex_adapter_instance_params(self):
        from askd.adapters.codex import CodexAdapter
        self._assert_adapter_has_instance_params(CodexAdapter)

    def test_gemini_adapter_instance_params(self):
        from askd.adapters.gemini import GeminiAdapter
        self._assert_adapter_has_instance_params(GeminiAdapter)

    def test_opencode_adapter_instance_params(self):
        from askd.adapters.opencode import OpenCodeAdapter
        self._assert_adapter_has_instance_params(OpenCodeAdapter)

    def test_claude_adapter_instance_params(self):
        from askd.adapters.claude import ClaudeAdapter
        self._assert_adapter_has_instance_params(ClaudeAdapter)

    def test_droid_adapter_instance_params(self):
        from askd.adapters.droid import DroidAdapter
        self._assert_adapter_has_instance_params(DroidAdapter)

    def test_copilot_adapter_instance_params(self):
        from askd.adapters.copilot import CopilotAdapter
        self._assert_adapter_has_instance_params(CopilotAdapter)

    def test_codebuddy_adapter_instance_params(self):
        from askd.adapters.codebuddy import CodebuddyAdapter
        self._assert_adapter_has_instance_params(CodebuddyAdapter)

    def test_qwen_adapter_instance_params(self):
        from askd.adapters.qwen import QwenAdapter
        self._assert_adapter_has_instance_params(QwenAdapter)


# ── Ask script qualified provider parsing ────────────────────────────────

class TestAskScriptQualifiedProvider:
    """Test that the ask script's provider parsing works for qualified names."""

    def test_ask_script_parses_qualified_provider(self):
        from providers import parse_qualified_provider
        # The ask script uses parse_qualified_provider internally
        base, inst = parse_qualified_provider("codex:auth")
        assert base == "codex"
        assert inst == "auth"


# ── Pane registry instance-qualified lookup ──────────────────────────────

class TestPaneRegistryInstance:
    """Test pane_registry integration with instance-qualified providers."""

    def test_pane_registry_imports_parse_qualified_provider(self):
        """Verify pane_registry can import and use parse_qualified_provider."""
        from providers import parse_qualified_provider
        # pane_registry uses this for provider lookups
        base, inst = parse_qualified_provider("gemini:frontend")
        assert base == "gemini"
        assert inst == "frontend"
