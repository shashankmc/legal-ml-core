"""Query strategies: the rules that rank the pool.

Each returns one score per pool item, and a higher score is served first. They
work on ``predict_proba`` output with NumPy, plus scikit-learn ``KMeans``.
"""
import numpy as np
from sklearn.cluster import KMeans

DEFAULT_STRATEGY = "prediction_entropy"


class UnknownStrategy(ValueError):
    pass


def _least_confidence(probabilities):
    return 1.0 - probabilities.max(axis=1)


def _prediction_entropy(probabilities):
    logs = np.log(probabilities, out=np.zeros_like(probabilities), where=probabilities > 0)
    return -(probabilities * logs).sum(axis=1) + 0.0


def _breaking_ties(probabilities):
    if probabilities.shape[1] < 2:
        return -probabilities[:, 0]
    top_two = np.sort(probabilities, axis=1)[:, -2:]
    return top_two[:, 0] - top_two[:, 1]


def _random(size, seed):
    return np.random.default_rng(seed).random(size)


def _embedding_kmeans(embeddings, batch_size, seed):
    """Spread a batch over the pool: every cluster's closest item, then every
    cluster's second closest, and so on."""
    size = embeddings.shape[0]
    clusters = min(batch_size, size)
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(embeddings)
    distances = kmeans.transform(embeddings)[np.arange(size), kmeans.labels_]

    rank_in_cluster = np.zeros(size, dtype=int)
    for cluster in range(clusters):
        members = np.flatnonzero(kmeans.labels_ == cluster)
        rank_in_cluster[members[np.argsort(distances[members])]] = np.arange(len(members))

    order = np.lexsort((distances, rank_in_cluster))
    scores = np.zeros(size)
    scores[order] = np.arange(size, 0, -1) / size
    return scores


STRATEGIES = {
    "least_confidence": _least_confidence,
    "prediction_entropy": _prediction_entropy,
    "breaking_ties": _breaking_ties,
    "embedding_kmeans": _embedding_kmeans,
    "random": _random,
}


def score_pool(strategy, probabilities=None, embeddings=None, batch_size=20, size=None, seed=42):
    """Score the pool with ``strategy``. Each needs a different input: class
    probabilities, embeddings and a batch size, or only the pool's ``size``."""
    if strategy not in STRATEGIES:
        raise UnknownStrategy(strategy)
    if strategy == "random":
        return _random(size, seed)
    if strategy == "embedding_kmeans":
        if embeddings.shape[0] == 0:
            return np.empty(0)
        return _embedding_kmeans(embeddings, batch_size, seed)
    if probabilities.shape[0] == 0:
        return np.empty(0)
    return STRATEGIES[strategy](probabilities)
