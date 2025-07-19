## 🔄 DWACE Pipeline Documentation
### DeepWalk AutoEncoder Contrastive Enhancement

## 📊 Pipeline Flow Diagram

```
Input Graph G
     ↓
📊 Step 1: Base Embedding Generation
     ├── DeepWalk Embedding (dim=128)
     └── Node2Vec Embedding (dim=128)
     ↓
🔧 Step 2: AutoEncoder Processing (if needed)
     ↓ (if embedding_dim > target_dim)
   AutoEncoder Dimensionality Reduction
   - Graph-aware loss (modularity + neighborhood)
   - Supervised classifier head (if ground truth)
     ↓
   deepwalk_ae
     ↓
🔀 Step 3: Enhancement Branch Selection
     ↓
   Ground Truth Available?
     ↓
   ┌─────────────────┴─────────────────┐
   ✅ YES                           ❌ NO
   (Karate, Football)               (Dolphins, Email)
   ↓                                ↓
📊 Contrastive Learning Branch      🔧 VAE Enhancement Branch
   ├── Data Augmentation            ├── Denoising AutoEncoder
   │   - Feature Dropout (10-15%)   │   - Gaussian Noise (σ=0.01)
   │   - Gaussian Noise (σ=0.05)    │   - L2 Regularization (λ=0.01)
   │   - Feature Shuffling (5%)     │   - Tanh Bottleneck
   ├── InfoNCE Loss                 ├── Reconstruction Loss
   ├── Supervised Loss              ├── Regularization Loss
   └── Gradient Clipping            └── Noise Robustness
   ↓                                ↓
deepwalk_ae_contrast              deepwalk_ae_vae
   ↓                                ↓
   └─────────────────┬─────────────────┘
                     ↓
              🎯 GAE-GCN Clustering
                     ↓
              📈 Performance Metrics
```

## 🎯 Branch Selection Logic

### ✅ Contrastive Learning Branch (Supervised)
**Used when**: Ground truth labels available
**Datasets**: Karate Club, Football
**Key Features**:
- Multi-view contrastive learning
- InfoNCE loss with improved numerical stability
- Supervised classification loss
- Advanced data augmentation
- Learning rate scheduling
- Gradient clipping

**Architecture**:
```
Input → Dense(256) → BatchNorm → Dropout(0.3) →
Dense(128) → BatchNorm → Dropout(0.2) →
Dense(64) → LayerNorm → [Projection Head | Classifier]
```

### ❌ VAE Enhancement Branch (Unsupervised)
**Used when**: No ground truth available
**Datasets**: Dolphins, Email, Facebook, etc.
**Key Features**:
- Denoising autoencoder approach
- L2 regularization on bottleneck
- Bounded representations (Tanh activation)
- Noise injection during training
- Reconstruction + regularization loss

**Architecture**:
```
Input → Dense(256) → BatchNorm → Dropout(0.3) →
Dense(128) → BatchNorm → Dropout(0.2) →
Dense(64, tanh) → Dense(128) → BatchNorm →
Dense(256) → Dense(input_dim)
```

## 🔧 Implementation Details

### Hyperparameters

**Contrastive Learning**:
- Temperature: 0.05
- Lambda contrastive: 1.0
- Lambda supervised: 0.3
- Epochs: 150
- Learning rate: 0.001 → decay 5% per 10 epochs

**VAE Enhancement**:
- L2 regularization: 0.01
- Noise std: 0.01
- Epochs: 80
- Learning rate: 0.001

**AutoEncoder (Base)**:
- Hidden dim: max(out_dim * 2, 32)
- Lambda reconstruction: 1.0
- Lambda modularity: 0.1
- Lambda neighborhood: 0.1
- Lambda supervised: 1.0

### Data Augmentation (Contrastive Branch)
1. **Feature Dropout**: 10-15% random feature masking
2. **Gaussian Noise**: σ = 0.05-0.08
3. **Feature Shuffling**: 5% of features, 30% probability

## 📈 Performance Comparison

### Karate Club (Supervised)
- **deepwalk_ae_contrast** performs best
- ARI: ~0.60, NMI: ~0.67
- Leverages ground truth effectively

### Dolphins (Unsupervised)
- **deepwalk_ae** and **deepwalk_ae_vae** competitive
- **deepwalk_ae_vae** shows consistent performance
- Modularity: ~0.48-0.50

## 🚀 Usage Example

```python
# The pipeline automatically selects the appropriate branch
embeddings_dict, enhancement_name = dwace_pipeline(
    G=graph,
    ground_truth=labels,  # None for unsupervised
    feature_dim=128,
    walk_params={'num_walks': 20, 'walk_length': 40}
)

# Results in either:
# enhancement_name = "deepwalk_ae_contrast" (supervised)
# enhancement_name = "deepwalk_ae_vae" (unsupervised)
```

## 🎯 Key Benefits

1. **Adaptive**: Automatically selects optimal enhancement strategy
2. **Robust**: Works well for both supervised and unsupervised scenarios  
3. **Scalable**: Efficient implementation with memory management
4. **Consistent**: Unified interface regardless of data type
5. **State-of-the-art**: Incorporates latest advances in contrastive learning and VAE
