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
