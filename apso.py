import numpy as np
from particle import AdaptiveParticle
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

class AdaptiveParticleSwarmOptimizedClustering:
    def __init__(
            self,
            n_cluster: int,
            n_particles: int,
            data: np.ndarray,
            hybrid: bool = False,
            max_iter: int = 1000,
            print_debug: int = 10,
            w_min: float = 0.4,
            w_max: float = 0.9,
            c1_min: float = 0.5,
            c1_max: float = 2.8,
            c2_min: float = 0.5,
            c2_max: float = 2.8,
            stagnation_threshold: int = 50):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid
        self.print_debug = print_debug
        self.gbest_score = np.inf
        self.gbest_centroids = None
        self.gbest_sse = np.inf
        self.w_min, self.w_max = w_min, w_max
        self.c1_min, self.c1_max = c1_min, c1_max
        self.c2_min, self.c2_max = c2_min, c2_max
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self._init_particles()

    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            if i == 0 and self.hybrid:
                particle = AdaptiveParticle(self.n_cluster, self.data, use_kmeans=True,
                                            w_min=self.w_min, w_max=self.w_max,
                                            c1_min=self.c1_min, c1_max=self.c1_max,
                                            c2_min=self.c2_min, c2_max=self.c2_max)
            else:
                particle = AdaptiveParticle(self.n_cluster, self.data, use_kmeans=False,
                                            w_min=self.w_min, w_max=self.w_max,
                                            c1_min=self.c1_min, c1_max=self.c1_max,
                                            c2_min=self.c2_min, c2_max=self.c2_max)
            if particle.best_score < self.gbest_score:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_score = particle.best_score
            self.particles.append(particle)
            self.gbest_sse = min(particle.best_sse, self.gbest_sse)

    def _evaluate_particle(self, particle):
        labels = particle._predict(self.data)
        n_clusters = len(np.unique(labels))
        if n_clusters < self.n_cluster:
            return np.inf  # Penalize solutions with fewer clusters
        
        error = quantization_error(particle.centroids, labels, self.data)
        return error

    def _reinitialize_particles(self):
        for particle in self.particles:
            particle.reinitialize(self.data, self.n_cluster)
        self._update_global_best()

    def _update_global_best(self):
        for particle in self.particles:
            score = self._evaluate_particle(particle)
            if score < self.gbest_score:
                self.gbest_centroids = particle.best_position.copy()
                self.gbest_score = score

    def run(self):
        #print('Initial global best score', self.gbest_score)
        self.history = []  # Initialize history as an attribute
        for i in range(self.max_iter):
            previous_best = self.gbest_score
            for particle in self.particles:
                particle.update(self.gbest_centroids, self.data)
                particle.update_parameters(i, self.max_iter)
                
                score = self._evaluate_particle(particle)
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.centroids.copy()
                
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_centroids = particle.centroids.copy()
            
            if self.gbest_score < previous_best:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            if self.stagnation_counter >= self.stagnation_threshold:
                #print(f"Stagnation detected at iteration {i+1}. Reinitializing particles.")
                self._reinitialize_particles()
                self.stagnation_counter = 0
            
            self.history.append(self.gbest_score)  # Append to self.history instead of local variable
            # if i % self.print_debug == 0:
            #     print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(i + 1, self.max_iter, self.gbest_score))
        #print('Finish with gbest score {:.18f}'.format(self.gbest_score))
        return self.history  # Return self.history instead of local variable