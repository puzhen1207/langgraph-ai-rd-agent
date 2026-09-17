"""Request guards: the admin API key and the anonymous client identity."""
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.core.logging_config import get_logger
from app.memory.session_registry import get_session_registry

logger = get_logger(__name__)


async def require_admin_key(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> None:
    """Require X-Admin-Key when ADMIN_API_KEY is configured.

    Local development remains convenient when the setting is empty. Production
    deployments should always configure a strong value.
    """
    expected = settings.ADMIN_API_KEY.strip()
    if not expected:
        return
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin API key",
        )


async def get_client_token(
    x_client_token: Optional[str] = Header(default=None, alias="X-Client-Token"),
) -> Optional[str]:
    """Read the anonymous token that scopes a browser to its own sessions."""
    if not x_client_token:
        return None
    return x_client_token.strip() or None


def claim_session(session_id: str, token: Optional[str]) -> None:
    """Authorise a chat request, binding a new session to the caller."""
    if get_session_registry().claim(session_id, token):
        return
    logger.warning("session_owner_mismatch", session_id=session_id)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Session belongs to another client",
    )


def assert_session_owner(session_id: str, token: Optional[str]) -> None:
    """Reject reads and deletes of a session owned by a different client."""
    if get_session_registry().is_owner(session_id, token):
        return
    logger.warning("session_access_denied", session_id=session_id)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Session belongs to another client",
    )
