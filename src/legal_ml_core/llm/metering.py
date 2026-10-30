"""Shared-key token metering.

``USAGE_METER`` is injected by the host. The default counts tokens per owner per
month in the database. A host can replace it to meter elsewhere.
"""

from datetime import datetime
from typing import Optional, Protocol

from flask import current_app
from sqlalchemy.exc import IntegrityError

from ..db import db
from ..models import LLMUsage
from .config import int_setting


class UsageMeter(Protocol):
    def remaining(self, owner_id: str) -> Optional[int]:
        """Tokens left this month, or ``None`` for no limit."""
        ...

    def record(self, owner_id: str, tokens: int, idempotency_key: str) -> None:
        """Add actual tokens. A repeated key must not count twice."""
        ...


class DBUsageMeter:
    """``idempotency_key`` is unused: record runs once, in the calling process."""

    @staticmethod
    def _period():
        return datetime.utcnow().strftime("%Y-%m")

    def remaining(self, owner_id):
        row = LLMUsage.query.filter_by(owner_id=owner_id, period=self._period()).first()
        used = row.tokens if row else 0
        return max(0, int_setting("LLM_MONTHLY_TOKEN_CAP") - used)

    def record(self, owner_id, tokens, idempotency_key):
        period = self._period()
        for _ in range(2):
            updated = LLMUsage.query.filter_by(owner_id=owner_id, period=period).update(
                {LLMUsage.tokens: LLMUsage.tokens + tokens}
            )
            if updated:
                db.session.commit()
                return
            db.session.add(LLMUsage(owner_id=owner_id, period=period, tokens=tokens))
            try:
                db.session.commit()
                return
            except IntegrityError:
                # Another process created this month's row first; update it.
                db.session.rollback()
        raise RuntimeError("Could not record LLM usage.")


def usage_meter():
    return current_app.config.get("USAGE_METER") or DBUsageMeter()
