"""Prototype auth: one shared bearer secret for the whole hackathon build.

Known simplification, NOT a production pattern -- there is no per-user identity
and no way to revoke a single caretaker. `verify_patient_scope` is the seam
where real per-user scoping drops in.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import config

_bearer = HTTPBearer(auto_error=False)


def require_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if not config.AUTH_SECRET:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "AUTH_SECRET unset")
    if not creds or not secrets.compare_digest(creds.credentials, config.AUTH_SECRET):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    return creds.credentials


def verify_patient_scope(patient_id: str, token: str) -> str:
    # ponytail: shared secret -> every holder can reach every patient. When real
    # auth lands, resolve token -> allowed patient ids here and 403 on mismatch.
    return patient_id
