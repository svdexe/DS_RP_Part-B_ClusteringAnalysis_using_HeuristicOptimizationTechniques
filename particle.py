import numpy as np
from kmeans import KMeans, calc_sse

def quantization_error(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray) -> float:
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)[0]
        if len(idx) > 0:
            dist = np.linalg.norm(data[idx] - c, axis=1).sum()
            dist /= len(idx)
            error += dist
    error /= len(centroids)
    return error

class Particle:
    def __init__(self,
                 n_cluster: int,
                 data: np.ndarray,
                 use_kmeans: bool = False,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        self.n_cluster = n_cluster
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        if use_kmeans:
            kmeans = KMeans(n_cluster=n_cluster, init_pp=False)
            kmeans.fit(data)
            self.centroids = kmeans.centroid.copy()
        self.best_position = self.centroids.copy()
        self.best_score = quantization_error(self.centroids, self._predict(data), data)
        self.best_sse = calc_sse(self.centroids, self._predict(data), data)
        self.velocity = np.zeros_like(self.centroids)
        self._w = w
        self._c1 = c1
        self._c2 = c2

    def update(self, gbest_position: np.ndarray, data: np.ndarray):
        self._update_velocity(gbest_position)
        self._update_centroids(data)

    def _update_velocity(self, gbest_position: np.ndarray):
        v_old = self._w * self.velocity
        cognitive_component = self._c1 * np.random.random() * (self.best_position - self.centroids)
        social_component = self._c2 * np.random.random() * (gbest_position - self.centroids)
        self.velocity = v_old + cognitive_component + social_component

    def _update_centroids(self, data: np.ndarray):
        self.centroids = self.centroids + self.velocity
        new_score = quantization_error(self.centroids, self._predict(data), data)
        sse = calc_sse(self.centroids, self._predict(data), data)
        self.best_sse = min(sse, self.best_sse)
        if new_score < self.best_score:
            self.best_score = new_score
            self.best_position = self.centroids.copy()

    def _predict(self, data: np.ndarray) -> np.ndarray:
        distance = self._calc_distance(data)
        cluster = self._assign_cluster(distance)
        return cluster

    def _calc_distance(self, data: np.ndarray) -> np.ndarray:
        distances = []
        for c in self.centroids:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    def _assign_cluster(self, distance: np.ndarray) -> np.ndarray:
        cluster = np.argmin(distance, axis=1)
        return cluster

class AdaptiveParticle(Particle):
    def __init__(self, n_cluster, data, use_kmeans=False, w_min=0.2, w_max=0.9, c1_min=0.5, c1_max=3, c2_min=0.5, c2_max=3):
        super().__init__(n_cluster, data, use_kmeans, w_max, c1_max, c2_min)
        self.w_min, self.w_max = w_min, w_max
        self.c1_min, self.c1_max = c1_min, c1_max
        self.c2_min, self.c2_max = c2_min, c2_max

    def update_parameters(self, iteration, max_iter):
        progress = (iteration / max_iter) ** 1.25  # Slightly non-linear progression
        self._w = self.w_max - (self.w_max - self.w_min) * progress
        self._c1 = self.c1_max - (self.c1_max - self.c1_min) * progress
        self._c2 = self.c2_min + (self.c2_max - self.c2_min) * progress

    def reinitialize(self, data):
        index = np.random.choice(list(range(len(data))), self.n_cluster)
        self.centroids = data[index].copy()
        self.velocity = np.random.uniform(-0.1, 0.1, size=self.centroids.shape)  # Small random velocity
        self.best_score = quantization_error(self.centroids, self._predict(data), data)
        self.best_sse = calc_sse(self.centroids, self._predict(data), data)
        self.best_position = self.centroids.copy()

    def reinitialize(self, data, n_cluster):
        index = np.random.choice(list(range(len(data))), n_cluster, replace=False)
        self.centroids = data[index].copy()
        self.velocity = np.random.uniform(-0.1, 0.1, size=self.centroids.shape)  # Small random velocity
        self.best_score = np.inf
        self.best_position = self.centroids.copy()

if __name__ == "__main__":
    pass