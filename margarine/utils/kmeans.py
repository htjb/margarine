"""K-Means clustering utility functions."""

import jax
import jax.numpy as jnp


def kmeans(X: jnp.ndarray, k: int, num_iters: int = 100) -> jnp.ndarray:
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
