# ============ DWACE PIPELINE (DeepWalk AutoEncoder Contrastive Enhancement) ============

"""
🔄 DWACE Pipeline Diagram:

Input Graph G
     ↓
📊 DeepWalk Embedding Generation (dim=128)
     ↓
🔧 AutoEncoder Processing (if dim reduction needed)
     ↓
   deepwalk_ae
     ↓
🔀 Ground Truth Check
     ↓
   ┌─────────────────┴─────────────────┐
   ✅ Has Ground Truth              ❌ No Ground Truth
   (Karate, Football)               (Dolphins, Email)
   ↓                                ↓
📊 Contrastive Learning Branch      🔧 VAE Enhancement Branch  
   - Data Augmentation              - Denoising AutoEncoder
   - InfoNCE Loss                   - L2 Regularization  
   - Supervised Loss                - Tanh Bottleneck
   ↓                                ↓
deepwalk_ae_contrast              deepwalk_ae_vae
   ↓                                ↓
   └─────────────────┬─────────────────┘
                     ↓
              🎯 GAE-GCN Clustering

Key Components:
1. Base Embeddings: DeepWalk + Node2Vec
2. AutoEncoder: Dimensionality reduction with graph-aware loss
3. Contrastive Branch: For supervised datasets (with ground truth)
4. VAE Branch: For unsupervised datasets (no ground truth)
5. Final Clustering: Using enhanced embeddings
"""

def dwace_pipeline_explanation():
    """
    Explain DWACE Pipeline components and flow
    """
    print("""
🔄 DWACE (DeepWalk AutoEncoder Contrastive Enhancement) Pipeline

📊 FLOW DIAGRAM:
Graph → DeepWalk → [AutoEncoder?] → {Ground Truth?} → Enhancement → Clustering
                                      ↓           ↓
                               Contrastive    VAE Enhancement
                               Learning       (unsupervised)
                               (supervised)

🎯 BRANCH SELECTION LOGIC:
- ✅ Has Ground Truth (Karate, Football): 
  → Contrastive Learning Branch
  → Uses InfoNCE + Supervised Loss
  → Output: deepwalk_ae_contrast

- ❌ No Ground Truth (Dolphins, Email):
  → VAE Enhancement Branch  
  → Uses Denoising AutoEncoder
  → Output: deepwalk_ae_vae

📈 BENEFITS:
- Adaptive: Automatically selects best enhancement method
- Supervised datasets: Leverages label information
- Unsupervised datasets: Uses advanced regularization
- Consistent interface: Same pipeline, different branches
    """)
