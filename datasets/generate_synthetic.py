import networkx as nx
import numpy as np
import pickle

def generate_large_communities(n_nodes=500, n_communities=5, p_intra=0.3, p_inter=0.01):
    """
    Tạo một graph lớn với ground truth communities
    """
    # Tạo community structure
    community_sizes = [n_nodes // n_communities] * n_communities
    # Điều chỉnh để tổng = n_nodes
    for i in range(n_nodes % n_communities):
        community_sizes[i] += 1
    
    # Tạo graph với community structure
    G = nx.random_partition_graph(community_sizes, p_intra, p_inter, seed=42)
    
    # Tạo ground truth labels
    ground_truth = []
    node_to_community = {}
    community_id = 0
    
    for block_size in community_sizes:
        for _ in range(block_size):
            ground_truth.append(community_id)
        community_id += 1
    
    # Gán node attributes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['community'] = ground_truth[i]
    
    return G, ground_truth

def generate_polbooks_like(n_books=400, n_publishers=3):
    """
    Tạo political books network lớn hơn
    """
    # Tạo 3 nhóm chính: Liberal, Conservative, Neutral
    community_sizes = [n_books // n_publishers] * n_publishers
    for i in range(n_books % n_publishers):
        community_sizes[i] += 1
    
    # Tạo connections trong và ngoài community
    G = nx.random_partition_graph(community_sizes, 0.4, 0.05, seed=123)
    
    # Tạo ground truth
    ground_truth = []
    for i, size in enumerate(community_sizes):
        ground_truth.extend([i] * size)
    
    return G, ground_truth

if __name__ == "__main__":
    print("Generating synthetic datasets...")
    
    # Dataset 1: Large Communities (500 nodes, 5 communities)
    G1, gt1 = generate_large_communities(500, 5)
    nx.write_edgelist(G1, "large_communities_500.txt")
    with open("large_communities_500_gt.pkl", "wb") as f:
        pickle.dump(gt1, f)
    print(f"Generated Large Communities: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    
    # Dataset 2: Political Books (400 nodes, 3 communities)  
    G2, gt2 = generate_polbooks_like(400, 3)
    nx.write_edgelist(G2, "polbooks_400.txt")
    with open("polbooks_400_gt.pkl", "wb") as f:
        pickle.dump(gt2, f)
    print(f"Generated Political Books: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
    
    # Dataset 3: Medium Scale (200 nodes, 4 communities)
    G3, gt3 = generate_large_communities(200, 4, 0.35, 0.02)
    nx.write_edgelist(G3, "medium_communities_200.txt")
    with open("medium_communities_200_gt.pkl", "wb") as f:
        pickle.dump(gt3, f)
    print(f"Generated Medium Communities: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")
    
    print("All synthetic datasets generated!")
