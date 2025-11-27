"""K-Means clustering utility functions."""

import jax
import jax.numpy as jnp


def kmeans(
    X: jnp.ndarray, k: int | jnp.ndarray, num_iters: int = 100
) -> jnp.ndarray:
    """Performs K-Means clustering on the given data.

    Args:
        X: Input data of shape (n_samples, n_features).
        k: Number of clusters.
        num_iters: Number of iterations for the K-Means algorithm.

    Returns:
        jnp.ndarray: Cluster labels for each data point.
    """

    @jax.jit
    def distance(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Calculate Euclidean distance between two sets of points.

        Args:
            a: First set of points.
            b: Second set of points.

        Returns:
            jnp.ndarray: Euclidean distances.
        """
        return jnp.sum((a - b) ** 2, axis=-1)

    initial_centroids = jax.random.multivariate_normal(
        jax.random.PRNGKey(0),
        jnp.mean(X, axis=0),
        jnp.cov(X, rowvar=False),
        shape=(k,),
    )

    vmap_distance = jax.vmap(distance, in_axes=(0, 0))

    def clustering_step(centroids: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Performs a single step of the K-Means clustering algorithm.

        Args:
            centroids: Current centroids of shape (k, n_features).
            X: Input data of shape (n_samples, n_features).

        Returns:
            jnp.ndarray: Updated centroids and labels.
        """
        initial_centroids = jnp.tile(
            centroids[jnp.newaxis, :, :], (X.shape[0], 1, 1)
        )
        distances = vmap_distance(initial_centroids, X)

        labels = jnp.argmin(distances, axis=1)
        new_centroids = jnp.array(
            [jnp.mean(X[labels == i], axis=0) for i in range(k)]
        )
        return new_centroids, labels

    centroids = initial_centroids
    for _ in range(num_iters):
        centroids, labels = clustering_step(centroids, X)

    return labels


def silhouette_score(X: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Calculates the silhouette score for the clustering.

    Args:
        X: Input data of shape (n_samples, n_features).
        labels: Cluster labels for each data point.

    Returns:
        float: Silhouette score.
    """

    def average_distance(x: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate the average distance from point x to all points in X."""
        return jnp.mean((X - x) ** 2)

    vmapped_avg_distance = jax.vmap(average_distance, in_axes=(0, None))

    silhouette_scores = []
    for label in jnp.unique(labels):
        cluster_points = X[labels == label]
        cluster_distance = vmapped_avg_distance(cluster_points, cluster_points)
        seperation_distance = jnp.min(
            jnp.array(
                [
                    vmapped_avg_distance(
                        cluster_points, X[labels == other_label]
                    )
                    for other_label in jnp.unique(labels)
                    if other_label != label
                ]
            )
        )
        # score per point in cluster
        score = (seperation_distance - cluster_distance) / jnp.maximum(
            seperation_distance, cluster_distance
        )
        # score per cluster
        silhouette_scores.append(jnp.mean(score))

    return jnp.mean(jnp.array(silhouette_scores))
