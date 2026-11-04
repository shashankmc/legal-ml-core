"""The groups a training metric point belongs to, so charts draw like with like.

- `train`: values the algorithm reports while it learns (loss, learning rate).
- `eval`: scores on the evaluation split (`eval_*`).
- `system`: resource and throughput readings.
- `stage`: the run's milestones (`training_started`, `model_saved`, ...), events
  rather than values.
"""

METRIC_GROUPS = ("train", "eval", "system", "stage")

STAGE_MARKERS = frozenset(
    {
        "preprocessing_started",
        "preprocessing_done",
        "training_started",
        "training_done",
        "model_saved",
        "training_cancelled",
        "training_failed",
    }
)


def metric_group_for(metric_name):
    """The group a point gets from its name alone. Used when a writer does not
    say, and for rows stored before points carried a group."""
    if metric_name in STAGE_MARKERS:
        return "stage"
    if metric_name.startswith("eval_"):
        return "eval"
    return "train"


# What a point's `training_step` counts, so a chart can title its x-axis.
X_KINDS = ("step", "iteration", "training_examples")

LEARNING_CURVE_PREFIX = "learning_curve_"


def x_kind_for(metric_name, epoch, training_step):
    """What the point's `training_step` counts, from what was stored, so old
    rows need no migration. A learning curve's points count training examples.
    Logistic regression's loss by iteration is the only other classical point
    with a `training_step`, and the only one without an epoch. Everything else
    counts the trainer's steps."""
    if metric_name.startswith(LEARNING_CURVE_PREFIX):
        return "training_examples"
    if training_step is not None and epoch is None:
        return "iteration"
    return "step"


def unit_for(metric_name, group):
    """A resource reading's unit, from the end of its name (system_metrics.py
    and the fine-tuning speed): `MB`, `%`, `samples/s` or `s`. None for any
    other point, whose values have no unit."""
    if group != "system":
        return None
    if metric_name.endswith("_mb"):
        return "MB"
    if metric_name.endswith("_percent"):
        return "%"
    if metric_name.endswith("_per_second"):
        return f"{metric_name.removesuffix('_per_second')}/s"
    if metric_name.startswith("seconds_") or metric_name.endswith("_seconds"):
        return "s"
    return None
