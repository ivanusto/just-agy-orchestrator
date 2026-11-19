"""Primary-model detection and settings.json handling."""

import json
import os
import re
import tempfile
from typing import Any, Dict, Optional, Tuple

from .common import ALLOWED_PRIMARY_MODELS, DEFAULT_PRIMARY_MODEL, _env, _env_flag


SMALL_TIER_RE = re.compile(r"\b(flash|lite|mini|nano|haiku)\b", re.IGNORECASE)
# Frontier primaries: Gemini 4 Argon or Carbon Medium/High, Gemini 4 Neon at every level,
# Gemini 4 Hydrogen High/XHigh (agy 1.5.7+),
# the agy Orchestrator (Neon/Carbon/Argon, plans with Carbon/Argon and executes with Hydrogen internally),
# and Claude Opus/Sonnet (any version). Argon/Carbon Low, Hydrogen Low/Medium, Flash, GPT-OSS and
# every other model are not frontier.
FRONTIER_TOKENS = ("argon", "carbon", "neon", "hydrogen", "orchestrator", "claude", "opus", "sonnet")
ALLOWED_PRIMARY_TEXT = (
    "Gemini 4 Argon or Carbon (Medium or High), Gemini 4 Neon (any level), Gemini 4 Hydrogen (High or XHigh), "
    "agy Orchestrator, or Claude Opus/Sonnet"
)


def is_frontier_tier(model_name: str) -> bool:
    """True for Gemini 4 Argon/Carbon Medium/High, Gemini 4 Neon at every level, Gemini 4 Hydrogen High/XHigh, the agy Orchestrator
    and Claude Opus/Sonnet, by display name or id."""
    normalized = re.sub(r"[_\s]+", "-", (model_name or "").strip().lower())
    if not normalized or normalized in {"auto", "none", "inherit"}:
        return False
    words = re.sub(r"[^a-z0-9.]+", " ", normalized)
    if SMALL_TIER_RE.search(words):
        return False
    if "orchestrator" in words or "neon" in words:
        return True
    if "hydrogen" in words:
        return not re.search(r"\b(low|medium)\b", words)
    if "argon" in words or "carbon" in words:
        return not re.search(r"\blow\b", words)
    return any(token in words for token in ("claude", "opus", "sonnet"))


is_argon_tier = is_frontier_tier  # legacy alias

# agy Orchestrator (Carbon) hands execution to an internal executor after exit_plan_mode; hook payloads
# then report this model. Users cannot select it, so it is accepted as part of the Orchestrator.
ORCHESTRATOR_EXECUTOR_MODELS = frozenset({"gemini-flash-worker-agy"})


def is_orchestrator_executor(model_name: str) -> bool:
    return (model_name or "").strip().lower() in ORCHESTRATOR_EXECUTOR_MODELS


def model_key(model_name: Any) -> str:
    """Spelling-independent key: 'Gemini 4 Neon (XHigh)' and 'gemini-4-neon-xhigh' compare equal."""
    return "-".join(seg for seg in re.split(r"[^a-z0-9]+", str(model_name or "").lower()) if seg)


def primary_cost_advice(model_name: Any) -> Optional[str]:
    """One-line advisory for agy primaries measured slower and costlier on sequential work.

    XHigh / Extra High levels and the agy Orchestrator family (not its executor) get the advice;
    every other model returns None. JUST_AGY_PRIMARY_ADVICE=0 turns it off.
    """
    if _env("PRIMARY_ADVICE", default="1") == "0":
        return None
    name = str(model_name or "").strip()
    normalized = name.lower()
    if not normalized or is_orchestrator_executor(normalized) or not is_frontier_tier(normalized):
        return None
    segments = [seg for seg in re.split(r"[^a-z0-9]+", normalized) if seg]
    xhigh = "xhigh" in segments or ("extra" in segments and "high" in segments)
    if not (xhigh or "orchestrator" in segments):
        return None
    kind = "an XHigh" if xhigh else "an Orchestrator"
    return (
        f"[Just agy Orchestrator] Note: '{name}' is {kind} primary. In an isolated agy benchmark "
        "(2026-11-19, 7 sequential single-directory tasks x 3) Gemini 4 Neon XHigh passed 19/21 and "
        "Orchestrator (Neon) 20/21, both about 2.8x the token-equivalent cost of Neon High (21/21) and "
        "slower; Hydrogen High also passed 21/21 at about a quarter of Neon High's cost. Mention once to "
        "the user that the High level is usually the better default for this kind of work; do not change "
        "the model yourself. JUST_AGY_PRIMARY_ADVICE=0 silences this."
    )


def settings_path() -> str:
    override = os.environ.get("AGY_ANTIGRAVITY_SETTINGS", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.expanduser("~/.gemini/antigravity-cli/settings.json")


def _read_settings() -> Tuple[Optional[Dict[str, Any]], str]:
    path = settings_path()
    if not os.path.exists(path):
        return None, path
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data if isinstance(data, dict) else None), path
    except (OSError, json.JSONDecodeError):
        return None, path


def _write_settings_atomic(path: str, data: Dict[str, Any]) -> None:
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".settings.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            json.dump(data, tmp_f, indent=2)
            tmp_f.write("\n")
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def check_primary_model_settings() -> Tuple[str, str]:
    """Return (status, model) where status is one of:
    'ok' (frontier), 'warn' (non-frontier, not changed), 'pinned' (rewritten, opt-in),
    'unknown' (settings file missing/unreadable), 'silenced'.
    """
    preferred = _env("PRIMARY_MODEL").strip()
    target = preferred if (preferred in ALLOWED_PRIMARY_MODELS or is_frontier_tier(preferred)) else DEFAULT_PRIMARY_MODEL
    data, path = _read_settings()
    if data is None:
        return "unknown", target
    current = str(data.get("model") or "").strip()

    if _env_flag("ALLOW_NON_PRIMARY"):
        return "silenced", current or target
    if is_frontier_tier(current) and (not preferred or current == preferred):
        return "ok", current
    if not _env_flag("PIN_SETTINGS"):
        return "warn", current
    try:
        data["model"] = target
        _write_settings_atomic(path, data)
        return "pinned", target
    except Exception:
        return "warn", current


POLICY_NOTICE = (
    "Subagent pinning is active: send to `flash_lite` (`TypeName: \"flash-lite-scout\"`) only reads of "
    "files at paths you name (single-file or symbol inspection at a known path). On agy 1.5+ scouts have "
    "only read_file (no search, listing or shell), so file/symbol location searches, grep/glob, git/log "
    "checks, test runs, and edits go to `flash` (`TypeName: \"flash-worker\"`); "
    "always give scouts exact paths. "
    "Do sequential web/docs research and tracing yourself (search_web, read_file); delegate only "
    "independent work that can run in parallel, and do not wait on a single subagent for something a "
    "few of your own calls would finish. "
    "When the task asks for an output file, write it as soon as you have an evidence-backed answer, "
    "then refine; do not open new verification branches before the deliverable exists. "
    "Never use `Model: \"inherit\"` or `Model: \"pro\"` on subagents. "
    "Every subagent prompt must end with the four-line REPORT FORMAT from rules/AGENTS.md."
)
