from .auth import (
    AuthError,
    authenticate_request,
    auth_mode,
    current_user,
    get_jwt,
    get_jwt_identity,
    jwt_required,
    verify_bearer_token,
)
from .blueprint import init_ml_feature
from .db import MLBase, db, get_db, set_active_db
from .models import (
    ALConfig,
    ALRound,
    AnnotationTask,
    Dataset,
    LLMAcknowledgement,
    LLMCredential,
    LLMPrompt,
    LLMSuggestion,
    LLMUsage,
    MLProject,
    Model,
    ModelComparison,
    TrainingJob,
    TrainingMetricPoint,
)
from .llm import DBUsageMeter, UsageMeter, purge_owner_llm_data
from .ownership import (
    HostModelOwnerRef,
    KeycloakOwnerRef,
    NoOwnerRef,
    OwnerRef,
    owner_ref,
)

# The worker imports the training stack (torch, transformers, shap, lime, nltk),
# which only `legal-ml-core[compute]` installs. Loading it on first use keeps
# `import legal_ml_core` cheap for an API process without that extra.
def __getattr__(name):
    if name == "run_ml_worker":
        from .worker import run_ml_worker

        return run_ml_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AuthError",
    "authenticate_request",
    "auth_mode",
    "current_user",
    "db",
    "MLBase",
    "get_db",
    "set_active_db",
    "get_jwt",
    "get_jwt_identity",
    "init_ml_feature",
    "jwt_required",
    "owner_ref",
    "run_ml_worker",
    "verify_bearer_token",
    "HostModelOwnerRef",
    "KeycloakOwnerRef",
    "NoOwnerRef",
    "OwnerRef",
    "Dataset",
    "Model",
    "TrainingJob",
    "ModelComparison",
    "AnnotationTask",
    "ALConfig",
    "ALRound",
    "TrainingMetricPoint",
    "LLMAcknowledgement",
    "LLMCredential",
    "LLMPrompt",
    "LLMSuggestion",
    "LLMUsage",
    "DBUsageMeter",
    "UsageMeter",
    "purge_owner_llm_data",
]
