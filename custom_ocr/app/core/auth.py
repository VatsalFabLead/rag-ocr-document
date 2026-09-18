from typing import Optional
from fastapi import Header, Query, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.services.api_key_service import api_key_service, APIKeyRecord

# Optional HTTPBearer scheme for OpenAPI documentation
security_bearer = HTTPBearer(auto_error=False)

async def verify_custom_api_key(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    x_custom_api_key: Optional[str] = Header(None, alias="X-Custom-API-Key"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None, alias="api_key")
) -> APIKeyRecord:
    """
    Validates custom model API key supplied in:
    1. Bearer token ('Authorization: Bearer <KEY>')
    2. Header 'X-Custom-API-Key: <KEY>' or 'X-API-Key: <KEY>'
    3. Query parameter '?api_key=<KEY>'
    """
    token_candidate: Optional[str] = None

    if bearer and bearer.credentials:
        token_candidate = bearer.credentials.strip()
    elif x_custom_api_key:
        token_candidate = x_custom_api_key.strip()
    elif x_api_key:
        token_candidate = x_api_key.strip()
    elif api_key:
        token_candidate = api_key.strip()

    if not token_candidate:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Missing Custom Model API Key",
                "hint": "Provide key via 'Authorization: Bearer <YOUR_KEY>' header, 'X-Custom-API-Key' header, or '?api_key=' parameter.",
                "docs": "Access /api/keys to generate your permanently free custom model key."
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    record = api_key_service.validate_key(token_candidate)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid or revoked Custom Model API Key",
                "provided_key": token_candidate[:8] + "..." if len(token_candidate) > 8 else token_candidate,
                "hint": "Check /api/keys to view or generate active project keys."
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    return record

async def optional_custom_api_key(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    x_custom_api_key: Optional[str] = Header(None, alias="X-Custom-API-Key"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None, alias="api_key")
) -> Optional[APIKeyRecord]:
    """Optional validation that does not fail if no key provided."""
    token_candidate: Optional[str] = None

    if bearer and bearer.credentials:
        token_candidate = bearer.credentials.strip()
    elif x_custom_api_key:
        token_candidate = x_custom_api_key.strip()
    elif x_api_key:
        token_candidate = x_api_key.strip()
    elif api_key:
        token_candidate = api_key.strip()

    if not token_candidate:
        return None

    return api_key_service.validate_key(token_candidate)
