"""How a training run is monitored: how often it logs and evaluates, and which
metric points it keeps. The package's defaults, then the host's
(`LEGAL_ML_MONITORING`), then a run's own (`parameters.monitoring`).

Each setting is one entry in MONITORING_FIELDS: its label, the name refusals
use, its default, and the check that says what is wrong with a value."""
from copy import deepcopy
from fnmatch import fnmatchcase

from .metric_groups import STAGE_MARKERS


def _whole_number_above_zero(label, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return f"{label} must be a whole number above 0."
    return None


def _one_of(*options):
    def check(label, value):
        if value not in options:
            return f"{label} must be one of: {', '.join(options)}."
        return None

    return check


def _names(label, value):
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        return f"{label} must be a list of names."
    return None


def _true_or_false(label, value):
    if not isinstance(value, bool):
        return f"{label} must be true or false."
    return None


MONITORING_FIELDS = {
    "log_every_n_steps": {"label": "Log every N steps", "default": 10, "check": _whole_number_above_zero},
    "eval_strategy": {"label": "Evaluation", "default": "epoch", "check": _one_of("epoch", "steps")},
    # Read only when eval_strategy is "steps".
    "eval_steps": {"label": "Evaluate every N steps", "default": 100, "check": _whole_number_above_zero},
    # Name globs, e.g. "eval_*". A point is kept when it matches an include and
    # no exclude. Stage markers are always kept.
    "include": {"label": "Metrics to keep", "default": ["*"], "check": _names},
    "exclude": {"label": "Metrics to drop", "default": [], "check": _names},
    # Accepted now, recorded later (BAIGLE #93, #94).
    "learning_curve": {"label": "Learning curve", "default": False, "check": _true_or_false},
    "system_metrics": {"label": "Resource readings", "default": False, "check": _true_or_false},
}


def monitoring_refusal(monitoring):
    """Why these monitoring settings cannot be used, or None, in the shape of a
    parameter refusal: `field` is `monitoring.<setting>`."""
    if not isinstance(monitoring, dict):
        return {"error": "Monitoring settings must be a set of named settings.", "field": "monitoring"}
    for name, value in monitoring.items():
        field = MONITORING_FIELDS.get(name)
        if field is None:
            return {"error": f"There is no monitoring setting called {name}.", "field": f"monitoring.{name}"}
        refusal = field["check"](field["label"], value)
        if refusal:
            return {"error": refusal, "field": f"monitoring.{name}"}
    return None


def effective_monitoring(host=None, run=None):
    """The settings a run uses: the package's defaults, then the host's, then the
    run's own. Both are checked before they get here."""
    defaults = {name: field["default"] for name, field in MONITORING_FIELDS.items()}
    return deepcopy(defaults | dict(host or {}) | dict(run or {}))


def validate_host_monitoring(config):
    """Fail at start, not on a user's first run, on a bad `LEGAL_ML_MONITORING`."""
    refusal = monitoring_refusal(config.get("LEGAL_ML_MONITORING", {}))
    if refusal:
        raise ValueError(f"LEGAL_ML_MONITORING: {refusal['error']} ({refusal['field']})")
