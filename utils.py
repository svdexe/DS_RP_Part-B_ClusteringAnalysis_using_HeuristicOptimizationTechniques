import numpy as np
from sklearn.metrics import silhouette_score

def normalize(x: np.ndarray):
    """Scale to 0-1"""
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

def standardize(x: np.ndarray):
    """Scale to zero mean unit variance"""
    return (x - x.mean(axis=0)) / np.std(x, axis=0)

def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        distances += dist
    return distances

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

def evaluate_clustering(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
    silhouette = silhouette_score(data, labels)
    sse = calc_sse(centroids, labels, data)
    quantization = quantization_error(centroids, labels, data)
    return silhouette, sse, quantization