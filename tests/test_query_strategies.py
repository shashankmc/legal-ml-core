"""Query strategies rank the pool from predict_proba output: higher goes first."""
import numpy as np
import pytest

from legal_ml_core.active_learning.strategies import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    UnknownStrategy,
    score_pool,
)

# Clear, torn, and a three-way tie.
PROBABILITIES = np.array([[0.9, 0.1, 0.0], [0.5, 0.45, 0.05], [1 / 3, 1 / 3, 1 / 3]])


def test_the_five_v1_strategies_exist_and_entropy_is_the_default():
    assert set(STRATEGIES) == {
        "least_confidence",
        "prediction_entropy",
        "breaking_ties",
        "embedding_kmeans",
        "random",
    }
    assert DEFAULT_STRATEGY == "prediction_entropy"


def test_least_confidence_is_one_minus_the_top_probability():
    scores = score_pool("least_confidence", probabilities=PROBABILITIES)

    assert scores == pytest.approx([0.1, 0.5, 2 / 3])


def test_prediction_entropy_is_shannon_entropy():
    scores = score_pool("prediction_entropy", probabilities=np.array([[0.5, 0.5], [1.0, 0.0]]))

    assert scores == pytest.approx([0.6931471805599453, 0.0])


def test_entropy_ranks_the_most_uncertain_first():
    scores = score_pool("prediction_entropy", probabilities=PROBABILITIES)

    assert list(np.argsort(-scores)) == [2, 1, 0]


def test_breaking_ties_prefers_the_smallest_gap_between_the_top_two():
    scores = score_pool("breaking_ties", probabilities=PROBABILITIES)

    assert scores == pytest.approx([-0.8, -0.05, 0.0])


def test_breaking_ties_survives_a_model_that_only_knows_one_class():
    scores = score_pool("breaking_ties", probabilities=np.array([[1.0], [1.0]]))

    assert scores == pytest.approx([-1.0, -1.0])


def test_random_is_reproducible_from_its_seed_and_needs_no_model():
    first = score_pool("random", size=50, seed=7)
    second = score_pool("random", size=50, seed=7)
    other = score_pool("random", size=50, seed=8)

    assert list(first) == list(second)
    assert list(first) != list(other)
    assert len(set(first)) == 50


def test_kmeans_puts_one_item_from_each_cluster_before_any_second_item():
    # Two tight groups, three items each: a batch of two must take one of each.
    embeddings = np.array(
        [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [10.0, 10.0], [10.0, 10.1], [10.1, 10.0]]
    )

    scores = score_pool("embedding_kmeans", embeddings=embeddings, batch_size=2, seed=1)

    first_two = set(np.argsort(-scores)[:2].tolist())
    assert len(first_two & {0, 1, 2}) == 1
    assert len(first_two & {3, 4, 5}) == 1


def test_kmeans_handles_a_batch_larger_than_the_pool():
    scores = score_pool("embedding_kmeans", embeddings=np.eye(3), batch_size=20, seed=1)

    assert len(set(scores)) == 3


def test_an_empty_pool_scores_nothing():
    for strategy in ("least_confidence", "prediction_entropy", "breaking_ties"):
        assert len(score_pool(strategy, probabilities=np.empty((0, 2)))) == 0
    assert len(score_pool("embedding_kmeans", embeddings=np.empty((0, 2)), batch_size=2)) == 0
    assert len(score_pool("random", size=0)) == 0


def test_an_unknown_strategy_is_refused():
    with pytest.raises(UnknownStrategy):
        score_pool("sampling_mode", probabilities=PROBABILITIES)
