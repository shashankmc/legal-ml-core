"""LLM assistance for annotation.

The package holds the capability only. Policy stays with the host, which injects
``FEATURE_GATE`` (who may use the shared key) and ``USAGE_METER`` (how shared-key
tokens are counted). See docs/adr/0002 and 0003 in the BAIGLE repo.
"""

from .access import purge_owner_llm_data
from .metering import DBUsageMeter, UsageMeter

__all__ = ["DBUsageMeter", "UsageMeter", "purge_owner_llm_data"]
