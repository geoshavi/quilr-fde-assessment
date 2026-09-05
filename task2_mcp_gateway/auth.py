"""Bearer token -> role resolution for the MCP security gateway.

ASSUMPTION (documented in README): there is no real identity provider for this
assessment. Token -> role is a small, explicit, hardcoded in-memory mapping.
These are demo values only, not real secrets.
"""

from __future__ import annotations

TOKEN_ROLES: dict[str, str] = {
    "admin-demo-token": "admin",
    "viewer-demo-token": "viewer",
}


def resolve_role(authorization_header: str | None) -> str | None:
    """Return the role for a well-formed `Bearer <token>` header, else None.

    None covers every failure mode uniformly (missing header, wrong scheme,
    empty token, unknown token) -- the caller does not need to distinguish
    them to decide the response, and not distinguishing them avoids giving an
    attacker a free oracle for "does this token exist".
    """
    if not authorization_header:
        return None
    scheme, sep, token = authorization_header.partition(" ")
    if not sep or scheme != "Bearer" or not token:
        return None
    return TOKEN_ROLES.get(token)
