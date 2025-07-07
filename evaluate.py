from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import networkx as nx
import numpy as np

def compute_modularity(G, labels):
    communities = {}
    for idx, node in enumerate(G.nodes()):
        label = labels[idx]
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    return nx.algorithms.community.quality.modularity(G, communities.values())

def compute_ari_nmi(labels, ground_truth):
    ari = adjusted_rand_score(ground_truth, labels)
    nmi = normalized_mutual_info_score(ground_truth, labels)
    return ari, nmi

def find_best_k(embeddings, G, ground_truth=None, k_range=None):
    if k_range is None:
        k_range = range(2, min(10, embeddings.shape[0]//2))
    inertias = []
    all_labels = []
    best_k = 2
    best_score = -1
    best_labels = None
    for k in k_range:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        all_labels.append(labels)
        inertias.append(kmeans.inertia_)
        try:
            score = silhouette_score(embeddings, labels)
        except:
            score = 0
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    # Elbow method: tìm điểm gấp khúc lớn nhất
    inertias = np.array(inertias)
    if len(inertias) > 2:
        # Tính độ dốc giữa các điểm
        deltas = np.diff(inertias)
        elbow_idx = np.argmin(deltas) + 1  # +1 vì diff giảm 1 phần tử
        elbow_k = k_range[elbow_idx]
        elbow_labels = all_labels[elbow_idx]
    else:
        elbow_k = best_k
        elbow_labels = best_labels
    modularity = compute_modularity(G, best_labels)
    elbow_modularity = compute_modularity(G, elbow_labels)
    if ground_truth is not None:
        ari, nmi = compute_ari_nmi(best_labels, ground_truth)
    else:
        ari = nmi = None
    # Trả về cả kết quả elbow
    return best_k, modularity, ari, nmi, best_labels, elbow_k, elbow_modularity, elbow_labels, inertias