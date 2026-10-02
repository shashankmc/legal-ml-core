"""Keycloak/OIDC bearer-token authentication for legal-ml-core routes.

This module replaces the previous in-app `flask_jwt_extended` authentication.
It intentionally exposes the same two names the route modules already imported
(`jwt_required`, `get_jwt_identity`) so migrating a route file is a one-line
import change rather than a rewrite of every decorated view.

Tokens are minted by Keycloak. We validate signature, issuer, audience and
expiry against the realm's JWKS, then hand the verified claims to an
application-supplied *identity resolver* which maps the token to a local user
row (just-in-time provisioning). The resolved local user's id becomes the JWT
identity, so every existing `user_id` foreign key and ownership check keeps
working unchanged.

The resolver is injected by the host app because the `users` model belongs to
the host, not to this package:

    app.config["IDENTITY_RESOLVER"] = resolve_local_user  # payload -> User

`resolve_local_user` must return an object with an `id` attribute (or a plain
mapping with an `id` key). It is called on every authenticated request; hosts
are expected to cache nothing themselves because the row lookup is cheap and
Keycloak remains the source of truth for credentials.
"""

import threading

import jwt
from jwt import PyJWKClient
from flask import current_app, g, jsonify, request

# Algorithms Keycloak may use to sign realm tokens. RS256 is the default; the
# others are included so a realm reconfigured to ECDSA keeps working.
_ALLOWED_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

# PyJWKClient already caches keys; a small registry avoids rebuilding clients
# (and re-reading discovery config) on every request. Keyed by JWKS URL.
_clients = {}
_clients_lock = threading.Lock()


class AuthError(Exception):
    """Raised when a request cannot be authenticated."""

    def __init__(self, message, status_code=401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _jwks_client(uri):
    with _clients_lock:
        client = _clients.get(uri)
        if client is None:
            # cache_keys=True keeps the signing keys in memory across requests;
            # PyJWKClient refreshes them when a token's `kid` is unknown.
            client = PyJWKClient(uri, cache_keys=True, lifespan=300)
            _clients[uri] = client
        return client


def _jwks_uri(config):
    explicit = (config.get("AUTH_JWKS_URI") or "").strip()
    if explicit:
        return explicit
    issuer = (config.get("AUTH_ISSUER") or "").rstrip("/")
    if not issuer:
        raise AuthError("AUTH_ISSUER is not configured.", status_code=500)
    return f"{issuer}/protocol/openid-connect/certs"


def verify_bearer_token(token):
    """Validate a Keycloak access token and return its claims.

    Raises `AuthError` for any problem (missing/invalid token, bad signature,
    wrong issuer/audience, expired) so callers return a uniform 401/500.
    """
    config = current_app.config
    issuer = (config.get("AUTH_ISSUER") or "").rstrip("/")
    audience = config.get("AUTH_AUDIENCE")

    if not issuer:
        raise AuthError("AUTH_ISSUER is not configured.", status_code=500)

    try:
        signing_key = _jwks_client(_jwks_uri(config)).get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=_ALLOWED_ALGORITHMS,
            issuer=issuer,
            audience=audience or None,
            options={"verify_aud": bool(audience)},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Access token has expired.")
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid access token: {exc}")
    except Exception as exc:  # network/JWKS failures and anything unexpected
        raise AuthError(f"Could not verify access token: {exc}")

    return payload


def _bearer_token():
    header = request.headers.get("Authorization", "")
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise AuthError("A bearer access token is required.")
    return parts[1].strip()


def authenticate_request():
    """Authenticate the current request, resolve the local user, return it."""
    payload = verify_bearer_token(_bearer_token())

    if not payload.get("sub"):
        raise AuthError("Access token has no subject.")

    resolver = current_app.config.get("IDENTITY_RESOLVER")
    if resolver is None:
        raise AuthError(
            "No identity resolver is configured on the application.",
            status_code=500,
        )

    user = resolver(payload)
    if user is None:
        raise AuthError("Authenticated identity is not permitted.")

    g.current_user = user
    g.jwt_payload = payload
    g.jwt_identity = str(user["id"] if isinstance(user, dict) else user.id)
    return user


def jwt_required(*decorator_args, **decorator_kwargs):
    """Drop-in replacement for `flask_jwt_extended.jwt_required`.

    Only the no-argument form is used across the codebase; positional/keyword
    args are accepted and ignored so `@jwt_required()` keeps its familiar
    spelling. Supports both `@jwt_required` and `@jwt_required()`.
    """

    def decorator(fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                authenticate_request()
            except AuthError as exc:
                return jsonify({"error": exc.message}), exc.status_code
            return fn(*args, **kwargs)

        return wrapper

    # Bare `@jwt_required` usage.
    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator(decorator_args[0])
    return decorator


def get_jwt_identity():
    """Return the resolved local user id, matching flask_jwt_extended's shape."""
    return getattr(g, "jwt_identity", None)


def get_jwt():
    """Return the raw verified claims (for endpoints that need claims)."""
    return getattr(g, "jwt_payload", None)


def current_user():
    """Return the resolved local user object, if authenticated."""
    return getattr(g, "current_user", None)
