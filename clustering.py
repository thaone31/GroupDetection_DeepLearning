from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import networkx as nx

def compute_modularity(G, labels):
    communities = {}
    for idx, node in enumerate(G.nodes()):
        label = labels[idx]
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    return nx.algorithms.community.quality.modularity(G, communities.values())

def cluster_all(embeddings, G, n_clusters=2):
    results = {}


    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)
    results['KMeans'] = kmeans_labels

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, assign_labels='kmeans')
    spectral_labels = spectral.fit_predict(embeddings)
    results['Spectral'] = spectral_labels

    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(embeddings)
    results['Agglomerative'] = agg_labels

    # Đánh giá từng thuật toán
    evals = {}
    for method, labels in results.items():
        if len(set(labels)) > 1 and -1 not in set(labels):
            modularity = compute_modularity(G, labels)
            try:
                sil = silhouette_score(embeddings, labels)
            except Exception:
                sil = None
        else:
            modularity = None
            sil = None
        evals[method] = {'modularity': modularity, 'silhouette': sil, 'labels': labels}
    return evals