# Community Detection with Node Embedding Methods

This project provides a comprehensive experimental framework for evaluating node embedding methods for community detection on various graph datasets. The pipeline supports classic and modern (SOTA) embedding models, advanced autoencoder and contrastive learning techniques, and a rich set of clustering metrics.

## Features
- **Multiple Embedding Methods:**
  - DeepWalk
  - node2vec
  - DeepWalk + Autoencoder (AE)
  - DeepWalk + AE + Contrastive Projection (DeepWalk_AE_Contrast_DWACE)
  - Graph Attention Network (GAT)
- **Flexible Dataset Support:**
  - Built-in and real-world graph datasets (Karate, Dolphins, Football, Email, Facebook)
  - Automatic subgraph sampling for large datasets
- **Advanced Loss Functions:**
  - Laplacian regularization, modularity loss, neighborhood preservation loss, and supervised loss for autoencoder
  - InfoNCE contrastive loss and supervised loss for contrastive projection
- **Clustering and Evaluation:**
  - KMeans, Spectral, Agglomerative clustering
  - Metrics: ARI, NMI, Modularity, Silhouette, Conductance, Coverage
  - Results aggregated over multiple runs (mean, std, min, max)
- **GPU Support:**
  - Efficient training with TensorFlow GPU memory growth enabled
- **Documentation:**
  - English and Vietnamese explanations, block diagrams, and code comments

## Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare datasets:**
   Place all required datasets in the `datasets/` directory as specified in the code.
3. **Run experiments:**
   ```bash
   python main.py
   ```
   - Select the dataset when prompted.
   - The pipeline will run all embedding/model/clustering combinations for 20 runs and output statistical summaries.


## Comparison with SOTA Embeddings
The framework benchmarks DeepWalk_AE_Contrast against SOTA node embedding models (DeepWalk, node2vec, GAT) using the same pipeline and metrics, enabling fair and reproducible comparison.


## File Structure
- `main.py`: Main experiment pipeline
- `models/`: Embedding and model implementations
- `datasets/`: Graph datasets
- `requirements.txt`: Python dependencies

## Acknowledgements
- This project uses open-source libraries: NetworkX, TensorFlow, scikit-learn, pandas, etc.

## Contact
For questions or contributions, please contact the project maintainer.
