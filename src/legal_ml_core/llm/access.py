"""Who may call which LLM, and with which credential."""

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken
from flask import current_app

from ..db import db
from ..models import LLMAcknowledgement, LLMCredential, LLMUsage
from .client import OpenAICompatibleClient
from .config import setting, shared_endpoint

SHARED_KEY_FEATURE = "llm.shared_key"


class CredentialError(Exception):
    """A stored key can't be used, with a message that is safe to show."""


@dataclass
class Credential:
    mode: str  # "shared" | "byok"
    base_url: str
    api_key: Optional[str]
    model: str
    destination: str
    extra_body: dict = field(default_factory=dict)

    def client(self):
        return OpenAICompatibleClient(
            self.base_url, self.api_key, self.model, self.extra_body
        )


def _fernet():
    key = setting("LLM_ENCRYPTION_KEY")
    if not key:
        raise CredentialError("Storing API keys is not configured on this platform.")
    return Fernet(key)


def encrypt_key(api_key):
    return _fernet().encrypt(api_key.encode()).decode()


def decrypt_key(encrypted):
    try:
        return _fernet().decrypt(encrypted.encode()).decode()
    except InvalidToken:
        raise CredentialError("The stored API key can't be read. Please add it again.")


def _is_public(host):
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    return bool(addresses) and all(
        ipaddress.ip_address(a).is_global for a in addresses
    )


def validate_byok_url(base_url):
    """Stop a user-supplied URL from reaching internal services.

    Returns the hostname, or raises ``CredentialError``. It runs when the key is
    saved and again before each use, since DNS can change in between.
    """
    parsed = urlparse(base_url or "")
    host = parsed.hostname
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        raise CredentialError("The base URL must be a plain https URL.")

    allowed = setting("LLM_BYOK_ALLOWED_HOSTS")
    if allowed:
        if isinstance(allowed, str):
            allowed = [h.strip() for h in allowed.split(",")]
        if host not in allowed:
            raise CredentialError("This base URL is not allowed on this platform.")
    elif not _is_public(host):
        raise CredentialError("The base URL must point to a public host.")
    return host


def shared_key_allowed(claims):
    """Fails closed: needs a verified email and a host gate that says yes."""
    if not claims or claims.get("email_verified") is not True:
        return False
    gate = current_app.config.get("FEATURE_GATE")
    if gate is None:
        return False
    try:
        return bool(gate(SHARED_KEY_FEATURE, claims))
    except Exception:
        current_app.logger.exception("FEATURE_GATE failed; denying the shared key")
        return False


def _shared_credential():
    endpoint = shared_endpoint()
    if endpoint is None:
        return None
    return Credential(
        mode="shared",
        base_url=endpoint["base_url"],
        api_key=endpoint.get("api_key"),
        model=endpoint["model"],
        destination=endpoint["destination"],
        extra_body=endpoint.get("extra_body") or {},
    )


def _byok_credential(owner_id):
    row = LLMCredential.query.filter_by(owner_id=owner_id).first()
    if row is None:
        return None
    return Credential(
        mode="byok",
        base_url=row.base_url,
        api_key=decrypt_key(row.encrypted_key),
        model=row.model,
        destination=urlparse(row.base_url).hostname,
    )


def _checked(credential):
    if credential is not None and credential.mode == "byok":
        validate_byok_url(credential.base_url)
    return credential


def resolve_credential(owner_id, claims):
    """The credential for a request. A user's own key wins over the shared key."""
    byok = _byok_credential(owner_id)
    if byok is not None:
        return _checked(byok)
    return _shared_credential() if shared_key_allowed(claims) else None


def credential_for_mode(owner_id, mode):
    """The worker's lookup. The API decided the mode when it queued the job."""
    if mode == "shared":
        return _shared_credential()
    return _checked(_byok_credential(owner_id))


def acknowledgement(owner_id, scope_id, destination):
    return LLMAcknowledgement.query.filter_by(
        owner_id=owner_id, scope_id=scope_id, destination=destination
    ).first()


def purge_owner_llm_data(owner_id):
    """Remove what the LLM feature holds for an owner, for account erasure.

    Hosts call this from their own account deletion. Prompts and suggestions go
    with the owner's datasets.
    """
    for model in (LLMCredential, LLMAcknowledgement, LLMUsage):
        model.query.filter_by(owner_id=owner_id).delete()
    db.session.commit()
