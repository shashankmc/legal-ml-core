"""Bearer-token authentication for legal-ml-core routes.

Three modes, selected by ``AUTH_MODE``: ``keycloak`` (default, described
below), ``platform`` (host-signed HS256 tokens, see ``verify_platform_token``)
and ``none`` (single-tenant, explicit opt-in).

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

# AUTH_MODE="platform" accepts only HS256 tokens signed with the shared platform
# secret, and only short-lived ones (exp - iat, in seconds).
_PLATFORM_ALGORITHMS = ["HS256"]
PLATFORM_TOKEN_MAX_LIFETIME = 60

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


def platform_secret():
    """Return the platform signing secret, read fresh on every request.

    ``LEGAL_ML_PLATFORM_SECRET`` (the value) wins over
    ``LEGAL_ML_PLATFORM_SECRET_FILE`` (a path the host writes, e.g. a read-only
    mounted volume). Where the secret lives is the host's concern; reading it
    per request means the host may generate it after this service starts.

    A missing secret is a deployment problem, not a sign-in problem, so it
    raises a 503 that names where the secret was expected.
    """
    config = current_app.config
    value = (config.get("LEGAL_ML_PLATFORM_SECRET") or "").strip()
    if value:
        return value

    path = (config.get("LEGAL_ML_PLATFORM_SECRET_FILE") or "").strip()
    if not path:
        message = (
            "AUTH_MODE='platform' but no platform secret is configured: set "
            "LEGAL_ML_PLATFORM_SECRET or LEGAL_ML_PLATFORM_SECRET_FILE."
        )
        current_app.logger.error(message)
        raise AuthError(message, status_code=503)

    try:
        with open(path, encoding="utf-8") as fh:
            value = fh.read().strip()
    except OSError:
        value = ""
    if not value:
        message = f"platform secret not found at {path}"
        current_app.logger.error(message)
        raise AuthError(message, status_code=503)
    return value


def verify_platform_token(token, secret=None):
    """Validate a platform-signed (HS256) token and return its claims.

    Used by ``AUTH_MODE="platform"``: the host signs one short-lived token per
    forwarded request with a shared secret. Only HS256 is accepted, on a path
    separate from Keycloak verification (which never accepts HS256). Tokens
    must carry a string ``sub`` (the owner), ``iat`` and ``exp``, and may be
    valid for at most ``PLATFORM_TOKEN_MAX_LIFETIME`` seconds.
    """
    if secret is None:
        secret = platform_secret()
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=_PLATFORM_ALGORITHMS,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid platform token: {exc}")

    # Checked here rather than relying on the PyJWT version's `sub` validation.
    if not isinstance(payload.get("sub"), str) or not payload["sub"]:
        raise AuthError("Invalid platform token: subject must be a non-empty string.")
    if payload["exp"] - payload["iat"] > PLATFORM_TOKEN_MAX_LIFETIME:
        raise AuthError(
            "Invalid platform token: lifetime exceeds "
            f"{PLATFORM_TOKEN_MAX_LIFETIME}s."
        )
    return payload


def _bearer_token():
    header = request.headers.get("Authorization", "")
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise AuthError("A bearer access token is required.")
    return parts[1].strip()


def auth_mode():
    """Return the configured auth mode: ``"keycloak"`` (default), ``"platform"`` or ``"none"``."""
    return (current_app.config.get("AUTH_MODE") or "keycloak").strip().lower()


def _assert_no_auth_is_safe():
    """Refuse to start in AUTH_MODE=none unless it has been explicitly allowed.

    An unauthenticated ML API is a serious footgun: it exposes every endpoint,
    including training and data access. We require the operator to set
    ``LEGAL_ML_ALLOW_NO_AUTH=true`` as a deliberate acknowledgement, and we log
    a loud warning. Raising here (at first request) rather than silently
    degrading makes the failure impossible to miss.
    """
    if not current_app.config.get("LEGAL_ML_ALLOW_NO_AUTH"):
        raise AuthError(
            "AUTH_MODE='none' disables authentication but "
            "LEGAL_ML_ALLOW_NO_AUTH is not set. Set it to 'true' to "
            "acknowledge that the ML API will be unauthenticated.",
            status_code=500,
        )
    if not current_app.config.get("_LEGAL_ML_NO_AUTH_WARNED"):
        current_app.logger.warning(
            "legal-ml-core is running with AUTH_MODE='none' — authentication "
            "and per-owner scoping are disabled. Do not expose this deployment."
        )
        current_app.config["_LEGAL_ML_NO_AUTH_WARNED"] = True


def authenticate_request():
    """Authenticate the request and resolve the owner (and optional user).

    In the default ``keycloak`` mode the bearer token is verified and the owner
    id is produced by the configured ``OwnerRef`` strategy (ownership.py) —
    the Keycloak ``sub`` by default, so no local user table is required. A host
    that still wants to JIT-provision a local row can set ``IDENTITY_RESOLVER``;
    its result is exposed via ``current_user()``.

    In ``platform`` mode the host signs one short-lived HS256 token per request
    with ``LEGAL_ML_PLATFORM_SECRET`` (or the file at
    ``LEGAL_ML_PLATFORM_SECRET_FILE``); the token ``sub`` is the owner via the
    same sub-based path. A missing secret is a 503, a bad token a 401, and a
    shared-owner strategy (``NoOwnerRef``) is refused.

    In ``none`` mode (ML-only deployments) no token is required; the owner comes
    from the (typically ``NoOwnerRef``) strategy and every request shares it.
    """
    from .ownership import NoOwnerRef, owner_ref

    if auth_mode() == "none":
        _assert_no_auth_is_safe()
        try:
            owner_id = owner_ref().resolve_owner_id(None)
        except ValueError as exc:
            raise AuthError(str(exc))

        g.current_user = None
        g.jwt_payload = None
        g.jwt_identity = owner_id
        return None

    if auth_mode() == "platform":
        if isinstance(owner_ref(), NoOwnerRef):
            raise AuthError(
                "AUTH_MODE='platform' scopes data per token subject, but "
                "OWNER_REF is NoOwnerRef (one shared owner). Remove OWNER_REF "
                "or use KeycloakOwnerRef().",
                status_code=500,
            )
        # Secret first: a missing secret must surface as 503, never as 401.
        secret = platform_secret()
        try:
            payload = verify_platform_token(_bearer_token(), secret)
        except AuthError as exc:
            current_app.logger.warning("platform token rejected: %s", exc.message)
            raise
    else:
        payload = verify_bearer_token(_bearer_token())

    if not payload.get("sub"):
        raise AuthError("Access token has no subject.")

    resolver = current_app.config.get("IDENTITY_RESOLVER")
    user = resolver(payload) if resolver is not None else None

    try:
        owner_id = owner_ref().resolve_owner_id(payload)
    except ValueError as exc:
        raise AuthError(str(exc))

    g.current_user = user
    g.jwt_payload = payload
    g.jwt_identity = owner_id
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
    """Return the current owner id (opaque), matching flask_jwt_extended's shape.

    Historically this was the local user id; it is now the owner id produced by
    the configured OwnerRef strategy (Keycloak `sub` by default).
    """
    return getattr(g, "jwt_identity", None)


def get_jwt():
    """Return the raw verified claims (for endpoints that need claims)."""
    return getattr(g, "jwt_payload", None)


def current_user():
    """Return the resolved local user object, if authenticated."""
    return getattr(g, "current_user", None)
