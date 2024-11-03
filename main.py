import numpy as np
import pandas as pd

from pso import ParticleSwarmOptimizedClustering
from apso import AdaptiveParticleSwarmOptimizedClustering
from utils import normalize, evaluate_clustering
from kmeans import KMeans

def run_algorithm(algorithm, data, n_runs=20):
    results = {
        'silhouette': [],
        'sse': [],
        'quantization': [],
    }
    for _ in range(n_runs):
        algorithm_instance = algorithm(
            n_cluster=3, n_particles=10, data=data, max_iter=2000, print_debug=2000)
        algorithm_instance.run()
        kmeans = KMeans(n_cluster=3, init_pp=False, seed=2018)
        kmeans.centroid = algorithm_instance.gbest_centroids.copy()
        predicted = kmeans.predict(data)
        
        n_unique_labels = len(np.unique(predicted))
        if n_unique_labels < 2:
            print(f"Warning: Only {n_unique_labels} unique cluster(s) found. Skipping evaluation.")
            continue
        
        silhouette, sse, quantization = evaluate_clustering(data, predicted, algorithm_instance.gbest_centroids)
        results['silhouette'].append(silhouette)
        results['sse'].append(sse)
        results['quantization'].append(quantization)
    
    return results

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv('seed.txt', sep='\t', header=None)
    x = data.drop([7], axis=1)
    x = x.values
    x = normalize(x)

    # Run K-means++
    kmeans = KMeans(n_cluster=3, init_pp=True, seed=2018)
    kmeans_results = run_algorithm(kmeans, x)

    # Run PSO
    pso_results = run_algorithm(ParticleSwarmOptimizedClustering, x)

    # Run Hybrid PSO
    hybrid_pso_results = run_algorithm(lambda *args, **kwargs: ParticleSwarmOptimizedClustering(*args, **kwargs, hybrid=True), x)

    # Run APSO
    apso_results = run_algorithm(AdaptiveParticleSwarmOptimizedClustering, x)

    # Run Hybrid APSO
    hybrid_apso_results = run_algorithm(lambda *args, **kwargs: AdaptiveParticleSwarmOptimizedClustering(*args, **kwargs, hybrid=True), x)

    # Prepare benchmark results
    benchmark = {
        'method': ['K-Means++', 'PSO', 'PSO Hybrid', 'APSO', 'APSO Hybrid'],
        'sse_mean': [
            np.mean(kmeans_results['sse']),
            np.mean(pso_results['sse']),
            np.mean(hybrid_pso_results['sse']),
            np.mean(apso_results['sse']),
            np.mean(hybrid_apso_results['sse']),
        ],
        'sse_stdev': [
            np.std(kmeans_results['sse']),
            np.std(pso_results['sse']),
            np.std(hybrid_pso_results['sse']),
            np.std(apso_results['sse']),
            np.std(hybrid_apso_results['sse']),
        ],
        'silhouette_mean': [
            np.mean(kmeans_results['silhouette']),
            np.mean(pso_results['silhouette']),
            np.mean(hybrid_pso_results['silhouette']),
            np.mean(apso_results['silhouette']),
            np.mean(hybrid_apso_results['silhouette']),
        ],
        'silhouette_stdev': [
            np.std(kmeans_results['silhouette']),
            np.std(pso_results['silhouette']),
            np.std(hybrid_pso_results['silhouette']),
            np.std(apso_results['silhouette']),
            np.std(hybrid_apso_results['silhouette']),
        ],
        'quantization_mean': [
            np.mean(kmeans_results['quantization']),
            np.mean(pso_results['quantization']),
            np.mean(hybrid_pso_results['quantization']),
            np.mean(apso_results['quantization']),
            np.mean(hybrid_apso_results['quantization']),
        ],
        'quantization_stdev': [
            np.std(kmeans_results['quantization']),
            np.std(pso_results['quantization']),
            np.std(hybrid_pso_results['quantization']),
            np.std(apso_results['quantization']),
            np.std(hybrid_apso_results['quantization']),
        ],
    }

    # Create and save benchmark DataFrame
    benchmark_df = pd.DataFrame.from_dict(benchmark)
    print(benchmark_df)
    benchmark_df.to_csv('benchmark_results.csv', index=False)