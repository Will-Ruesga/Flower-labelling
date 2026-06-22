"""In-memory store for the single active ``RunContext``."""

from shared import RunContext


_context: RunContext | None = None


def get_context() -> RunContext:
    """Return the current context, initializing a default one on first access."""
    global _context
    if _context is None:
        _context = RunContext()
    return _context


def reset_context() -> None:
    """Discard the current context and install a fresh default one."""
    global _context
    _context = RunContext()
