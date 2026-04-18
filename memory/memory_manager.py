"""Persistent memory management for session history."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_FILE = Path("./memory/user_memory.json")
_MAX_SESSIONS = 12


def load_memory(
    filepath: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """
    Load session history from the JSON memory file.

    Returns an empty session store if the file does not exist, is empty,
    or contains invalid JSON — never raises an exception.

    Args:
        filepath: Path to the JSON memory file.
            Defaults to ``./memory/user_memory.json``.

    Returns:
        Dict with a ``"sessions"`` key whose value is a list of monthly
        summary dicts (most-recent first, up to 12 entries).

    Example:
        >>> memory = load_memory()
        >>> memory["sessions"]
        [{"month": "March", "year": 2026, ...}]
    """
    path = Path(filepath) if filepath is not None else DEFAULT_MEMORY_FILE
    logger.debug("load_memory() | path=%s", path)

    if not path.exists():
        logger.info("Memory file not found at '%s' — returning empty store", path)
        return {"sessions": []}

    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.info("Memory file '%s' is empty — returning empty store", path)
            return {"sessions": []}

        data = json.loads(text)

        if not isinstance(data, dict) or "sessions" not in data:
            logger.warning(
                "Invalid memory structure in '%s' — returning empty store", path
            )
            return {"sessions": []}

        sessions: List[Dict] = data["sessions"]
        if not isinstance(sessions, list):
            logger.warning("'sessions' is not a list in '%s' — resetting", path)
            return {"sessions": []}

        logger.info("Loaded %d session(s) from '%s'", len(sessions), path)
        return {"sessions": sessions}

    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error loading memory from '%s': %s", path, exc)
        return {"sessions": []}


def save_memory(
    session_summary: Dict[str, Any],
    filepath: Optional[Union[Path, str]] = None,
) -> bool:
    """
    Append a monthly session summary to the persistent memory file.

    Behaviour:
      1. Load existing sessions (graceful if file missing).
      2. Append the new session.
      3. Trim to the last ``_MAX_SESSIONS`` (12) entries.
      4. Back up the current file to ``<file>.backup``.
      5. Write the updated store atomically.

    Args:
        session_summary: Monthly summary dict to persist.
        filepath: Path to the JSON memory file.
            Defaults to ``./memory/user_memory.json``.

    Returns:
        ``True`` on success, ``False`` on any error (never raises).

    Example:
        >>> save_memory({"month": "March", "year": 2026, ...})
        True
    """
    if not isinstance(session_summary, dict) or not session_summary:
        logger.error("save_memory() received invalid session_summary — aborting")
        return False

    path = Path(filepath) if filepath is not None else DEFAULT_MEMORY_FILE
    logger.debug("save_memory() | path=%s", path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        memory = load_memory(path)
        sessions = memory["sessions"]

        # Upsert: replace existing entry for the same month+year, else append.
        new_month = session_summary.get("month")
        new_year  = session_summary.get("year")
        replaced  = False
        for i, s in enumerate(sessions):
            if s.get("month") == new_month and s.get("year") == new_year:
                sessions[i] = session_summary
                replaced = True
                logger.info("Updated existing session for %s %s", new_month, new_year)
                break
        if not replaced:
            sessions.append(session_summary)

        # Rolling 12-month window
        if len(sessions) > _MAX_SESSIONS:
            removed = len(sessions) - _MAX_SESSIONS
            memory["sessions"] = sessions[-_MAX_SESSIONS:]
            logger.warning(
                "Memory trimmed to %d sessions (removed %d oldest)",
                _MAX_SESSIONS,
                removed,
            )

        # Validate serialisation before touching the file
        serialised = json.dumps(memory, indent=2, ensure_ascii=False)

        # Backup existing file
        if path.exists():
            backup = path.with_suffix(path.suffix + ".backup")
            try:
                shutil.copy2(path, backup)
                logger.debug("Backup written to '%s'", backup)
            except OSError as bk_exc:
                logger.warning("Could not create backup '%s': %s", backup, bk_exc)

        path.write_text(serialised, encoding="utf-8")
        logger.info(
            "Session saved to '%s' (%d total)", path, len(memory["sessions"])
        )
        return True

    except Exception as exc:
        logger.error("Error saving memory to '%s': %s", path, exc)
        return False


def clear_memory(
    filepath: Optional[Union[Path, str]] = None,
) -> bool:
    """
    Reset the memory file to an empty session store.

    Creates a backup before clearing. Never raises — errors are logged
    and ``False`` is returned.

    Args:
        filepath: Path to the JSON memory file.
            Defaults to ``./memory/user_memory.json``.

    Returns:
        ``True`` on success, ``False`` on any error.

    Example:
        >>> clear_memory()
        True
    """
    path = Path(filepath) if filepath is not None else DEFAULT_MEMORY_FILE
    logger.warning("clear_memory() | path=%s", path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Backup before clearing
        if path.exists():
            backup = path.with_suffix(path.suffix + ".backup")
            try:
                shutil.copy2(path, backup)
                logger.debug("Backup written to '%s' before clear", backup)
            except OSError as bk_exc:
                logger.warning("Could not create backup before clear: %s", bk_exc)

        path.write_text(
            json.dumps({"sessions": []}, indent=2),
            encoding="utf-8",
        )
        logger.info("Memory cleared at '%s'", path)
        return True

    except Exception as exc:
        logger.error("Error clearing memory at '%s': %s", path, exc)
        return False
