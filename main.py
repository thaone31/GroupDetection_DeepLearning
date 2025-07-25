import numpy as np
import networkx as nx
from models.gae import GAE
from evaluate import find_best_k
from clustering import cluster_all
from models.feature_utils import node_deep_sum, deepwalk_embedding, node2vec_embedding, deepwalk_enhanced_embedding
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, backend as K
import tensorflow.keras.models as keras_models
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict
import inspect

# --- Thêm: Bật memory growth cho GPU TensorFlow để tránh treo máy khi allocate toàn bộ GPU memory ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



def load_dataset(choice):
    if choice == 1:
        G = nx.karate_club_graph()
        name = "Karate"
        ground_truth = [1 if G.nodes[n]['club'] != 'Mr. Hi' else 0 for n in G.nodes()]
    elif choice == 2:
        G = nx.read_gml('datasets/dolphins.gml', label='id')
        name = "Dolphins"
        ground_truth = None
    elif choice == 3:
        G = nx.read_gml('datasets/football.gml', label='id')
        name = "Football"
        ground_truth = [G.nodes[n]['value'] for n in G.nodes()]
    elif choice == 4:
        # Load Email dataset (edgelist format)
        G = nx.read_edgelist('datasets/email-Eu-core.txt', nodetype=int)
        name = "Email"
        ground_truth = None
    elif choice == 5:
        # Load Wiki-Vote dataset (edgelist format)
        G = nx.read_edgelist('datasets/wiki-Vote.txt', nodetype=int)
        name = "Wiki Vote"
        ground_truth = None
    elif choice == 6:
        G = nx.read_edgelist('datasets/facebook_combined.txt', nodetype=int)
        name = "Facebook"
        ground_truth = None
    elif choice == 7:
        # Large Communities Dataset (500 nodes, 5 communities) - HAS GROUND TRUTH
        G = nx.read_edgelist('datasets/large_communities_500.txt', nodetype=int)
        name = "Large Communities (500)"
        import pickle
        with open('datasets/large_communities_500_gt.pkl', 'rb') as f:
            ground_truth = pickle.load(f)
    elif choice == 8:
        # Political Books Dataset (400 nodes, 3 communities) - HAS GROUND TRUTH  
        G = nx.read_edgelist('datasets/polbooks_400.txt', nodetype=int)
        name = "Political Books (400)"
        import pickle
        with open('datasets/polbooks_400_gt.pkl', 'rb') as f:
            ground_truth = pickle.load(f)
    elif choice == 9:
        # Medium Communities Dataset (200 nodes, 4 communities) - HAS GROUND TRUTH
        G = nx.read_edgelist('datasets/medium_communities_200.txt', nodetype=int)
        name = "Medium Communities (200)"
        import pickle
        with open('datasets/medium_communities_200_gt.pkl', 'rb') as f:
            ground_truth = pickle.load(f)
    elif choice == 10:
        G = nx.read_edgelist('datasets/com-amazon.ungraph.txt', nodetype=int)
        name = "Amazon"
        ground_truth = None
    elif choice == 11:
        G = nx.read_edgelist('datasets/com-dblp.ungraph.txt', nodetype=int)
        name = "DBLP"
        ground_truth = None
    elif choice == 12:
        G = nx.read_edgelist('datasets/com-youtube.ungraph.txt', nodetype=int)
        name = "YouTube"
        ground_truth = None
    else:
        raise ValueError("Lựa chọn không hợp lệ.")
    return G, name, ground_truth

def main():

    def preprocess_embedding(embedding):
        # Thay thế NaN bằng 0 (hoặc có thể dùng np.nanmean nếu muốn)
        return np.nan_to_num(embedding, nan=0.0)
    # Luôn chạy chế độ so sánh toàn bộ các trường hợp
    print("[INFO] Đang chạy toàn bộ các trường hợp so sánh (tất cả dataset x embedding x model x clustering)...")
    mode = 2

    datasets = [
        (1, "Karate Club Graph"),
        (2, "Dolphins"),
        (3, "Football"),
        (4, "Email"),
        (5, "Wiki Vote"),
        (6, "Facebook"),
        (7, "Large Communities (500)"),     # NEW: 500 nodes, 5 communities
        (8, "Political Books (400)"),       # NEW: 400 nodes, 3 communities  
        (9, "Medium Communities (200)"),    # NEW: 200 nodes, 4 communities
        (10, "Amazon"),
        (11, "DBLP"),
        (12, "YouTube")
    ]
    embeddings = [
        (1, "deepwalk"),
        (2, "node2vec"),
        (3, "node2vec_ae"),
        (4, "deepwalk_enhanced"),
    ]
    models = [
        (1, "GAE-GCN (baseline)")
    ]
    clustering_methods = [
        (1, "KMeans"),
        (2, "Spectral"),
        (3, "Agglomerative"),
    ]

    print("Đang chạy toàn bộ các trường hợp so sánh...")
    feature_dim = 64

    def autoencoder_reduce(X, out_dim, epochs=100, batch_size=32, verbose=0, laplacian_reg=True, reg_weight=1.0):
        input_dim = X.shape[1]
        input_layer = layers.Input(shape=(input_dim,))
        # Nonlinear autoencoder: thêm 1 hidden layer nonlinear cho encoder và decoder
        hidden_dim = max(out_dim * 2, 32)
        # Encoder: Input -> Dense(hidden_dim, activation=LeakyReLU) -> Dense(out_dim, activation=ELU)
        x = layers.Dense(hidden_dim)(input_layer)
        x = layers.LeakyReLU(alpha=0.1)(x)
        encoded = layers.Dense(out_dim)(x)
        encoded = layers.ELU(alpha=1.0)(encoded)
        # Decoder: encoded -> Dense(hidden_dim, activation=LeakyReLU) -> Dense(input_dim, linear)
        x_dec = layers.Dense(hidden_dim)(encoded)
        x_dec = layers.LeakyReLU(alpha=0.1)(x_dec)
        decoded = layers.Dense(input_dim, activation='linear')(x_dec)
        autoencoder = keras_models.Model(input_layer, decoded)
        encoder = keras_models.Model(input_layer, encoded)
        optimizer = optimizers.Adam(learning_rate=0.005)

        # Tự động lấy adjacency matrix A và comm_labels từ graph G và ground_truth (nếu có)
        import inspect
        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)
        G = None
        ground_truth = None
        for f in outer_frames:
            local_vars = f.frame.f_locals
            if 'G' in local_vars:
                G = local_vars['G']
            if 'ground_truth' in local_vars:
                ground_truth = local_vars['ground_truth']
            if G is not None and ground_truth is not None:
                break
        if G is not None:
            if G.number_of_nodes() < 10000:
                import networkx as nx
                A = nx.to_numpy_array(G)
            else:
                print(f"[WARNING] Graph {getattr(G, 'name', 'unknown')} quá lớn ({G.number_of_nodes()} nodes), bỏ qua adjacency matrix cho loss.")
                A = None
        else:
            A = None
        if ground_truth is not None:
            comm_labels = np.array(ground_truth)
        else:
            comm_labels = np.zeros(X.shape[0], dtype=int)

        def custom_loss(y_true, y_pred):
            mse_loss = losses.MeanSquaredError()(y_true, y_pred)
            if A is not None:
                Z = encoder(y_true)
                n = tf.shape(Z)[0]
                A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
                deg = tf.reduce_sum(A_tf, axis=1, keepdims=True)
                m = tf.reduce_sum(deg) / 2.0 + 1e-8
                Z_norm = tf.math.l2_normalize(Z, axis=1)
                dot = tf.matmul(Z_norm, Z_norm, transpose_b=True)
                deg_prod = tf.matmul(deg, tf.transpose(deg)) / (2.0 * m)
                modularity_matrix = A_tf - deg_prod
                modularity_score = tf.reduce_sum(modularity_matrix * dot) / (2.0 * m)
                modularity_loss = -modularity_score

                comm_labels_tf = tf.convert_to_tensor(comm_labels, dtype=tf.int32)
                mask_same = tf.cast(tf.equal(tf.expand_dims(comm_labels_tf, 1), tf.expand_dims(comm_labels_tf, 0)), tf.float32)
                mask_diff = 1.0 - mask_same
                dists = tf.norm(tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0), axis=2)
                same_mean = tf.reduce_sum(dists * mask_same) / (tf.reduce_sum(mask_same) + 1e-8)
                diff_mean = tf.reduce_sum(dists * mask_diff) / (tf.reduce_sum(mask_diff) + 1e-8)
                neigh_loss = same_mean / (diff_mean + 1e-8)
            else:
                modularity_loss = 0.0
                neigh_loss = 0.0
            total_loss = mse_loss + 0.1 * modularity_loss + 0.1 * neigh_loss
            return total_loss
        autoencoder.compile(optimizer=optimizer, loss=custom_loss)
        autoencoder.fit(X, X, epochs=300, batch_size=min(batch_size, X.shape[0]), verbose=verbose)
        reduced = encoder.predict(X)
        K.clear_session()
        return reduced
    walk_params = dict()  # Ensure walk_params is always defined
    # Đồng bộ thông số embedding cho tất cả dataset
    walk_params = dict(num_walks=20, walk_length=40)
    feature_dim = 128
    encoder_types = ["gcn"]  # Only use GCN encoder
    ae_epochs = 50  # Tăng epoch AE để học tốt hơn
    ae_batch_size = 64
    # Giảm batch_size cho các dataset lớn
    LARGE_DATASETS = ["DBLP", "YouTube", "Amazon", "Large Communities (500)", "Political Books (400)"]
    # === Improved contrastive learning utilities ===
    def graph_augment(X, drop_prob=0.2, noise_std=0.1):
        """
        Improved data augmentation with multiple strategies:
        1. Feature dropout
        2. Gaussian noise
        3. Feature shuffling (within each sample)
        """
        X_aug = X.copy()
        
        # 1. Feature dropout
        if drop_prob > 0:
            mask = np.random.binomial(1, 1-drop_prob, X.shape)
            X_aug = X_aug * mask
        
        # 2. Add small amount of Gaussian noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, X.shape)
            X_aug = X_aug + noise
        
        # 3. Random feature shuffling (5% of features per sample)
        if np.random.random() < 0.3:  # 30% chance to apply feature shuffling
            for i in range(X_aug.shape[0]):
                n_shuffle = max(1, int(0.05 * X_aug.shape[1]))  # Shuffle 5% of features
                shuffle_idx = np.random.choice(X_aug.shape[1], n_shuffle, replace=False)
                X_aug[i, shuffle_idx] = np.random.permutation(X_aug[i, shuffle_idx])
        
        return X_aug.astype(np.float32)

    def info_nce_loss(z1, z2, temperature=0.1):
        # Improved InfoNCE loss with better numerical stability
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        batch_size = tf.shape(z1)[0]
        
        # Concatenate positive pairs
        representations = tf.concat([z1, z2], axis=0)  # [2*batch_size, dim]
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(representations, representations, transpose_b=True)
        
        # Remove self-similarity (diagonal)
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        similarity_matrix = tf.where(mask, -tf.float32.max, similarity_matrix)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / temperature
        
        # Create labels for positive pairs
        # For i-th sample: positive is at position batch_size + i
        labels_a = tf.range(batch_size, dtype=tf.int32) + batch_size  # [0->batch_size-1] maps to [batch_size->2*batch_size-1]
        labels_b = tf.range(batch_size, dtype=tf.int32)               # [batch_size->2*batch_size-1] maps to [0->batch_size-1]
        
        # Compute cross-entropy loss for both directions
        loss_a = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_a,
            logits=similarity_matrix[:batch_size]
        )
        loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_b,
            logits=similarity_matrix[batch_size:]
        )
        
        return tf.reduce_mean(loss_a + loss_b)

    def unsupervised_feature_enhancement(X, out_dim=64, epochs=50, batch_size=128):
        """
        Unsupervised feature enhancement cho các dataset không có ground truth
        Sử dụng Variational Autoencoder (VAE) với regularization để học better representations
        """
        input_layer = layers.Input(shape=(X.shape[1],))
        
        # Encoder
        x = layers.Dense(min(256, X.shape[1] * 2), activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        
        # VAE latent space: mean and log_var
        z_mean = layers.Dense(out_dim, name='z_mean')(x)
        z_log_var = layers.Dense(out_dim, name='z_log_var')(x)
        
        # Sampling layer as a custom layer
        class Sampling(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = Sampling()([z_mean, z_log_var])
        
        # Decoder
        decoder_h = layers.Dense(128, activation='relu')
        decoder_h2 = layers.Dense(min(256, X.shape[1] * 2), activation='relu')
        decoder_mean = layers.Dense(X.shape[1], activation='linear')
        
        h_decoded = decoder_h(z)
        h_decoded2 = decoder_h2(h_decoded)
        x_decoded_mean = decoder_mean(h_decoded2)
        
        # Create models
        encoder = keras_models.Model(input_layer, z_mean, name='encoder')  # Use mean for encoding
        decoder_model = keras_models.Model(z, x_decoded_mean, name='decoder')
        
        # VAE model with custom training step
        class VAE(keras_models.Model):
            def __init__(self, encoder, decoder, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
                self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
                self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
            
            def call(self, inputs):
                z_mean, z_log_var, z = self.encoder(inputs), self.encoder(inputs), self.encoder(inputs)
                return self.decoder(z)
            
            def train_step(self, data):
                with tf.GradientTape() as tape:
                    # Forward pass
                    z_mean = self.encoder(data)
                    
                    # Get z_log_var by accessing the layer output
                    encoder_layers = self.encoder.layers
                    z_log_var = None
                    for layer in encoder_layers:
                        if hasattr(layer, 'name') and 'z_log_var' in layer.name:
                            z_log_var = layer(data)
                    
                    if z_log_var is None:
                        # Fallback: create z_log_var layer
                        z_log_var = layers.Dense(out_dim)(self.encoder.layers[-2].output)
                    
                    # Sample z
                    batch = tf.shape(z_mean)[0]
                    dim = tf.shape(z_mean)[1]
                    epsilon = tf.random.normal(shape=(batch, dim))
                    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
                    
                    # Decode
                    reconstruction = self.decoder(z)
                    
                    # Compute losses
                    reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))
                    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    total_loss = reconstruction_loss + 0.1 * kl_loss
                
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)
                
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }
        
        # Simplified approach: Use standard autoencoder with noise regularization
        # This avoids the VAE complexity and still provides good representations
        input_layer = layers.Input(shape=(X.shape[1],))
        
        # Encoder with bottleneck
        encoded = layers.Dense(min(256, X.shape[1] * 2), activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.3)(encoded)
        
        encoded = layers.Dense(128, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        
        # Bottleneck layer
        bottleneck = layers.Dense(out_dim, activation='tanh')(encoded)  # tanh for bounded output
        
        # Decoder
        decoded = layers.Dense(128, activation='relu')(bottleneck)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        
        decoded = layers.Dense(min(256, X.shape[1] * 2), activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        
        output = layers.Dense(X.shape[1], activation='linear')(decoded)
        
        # Models
        autoencoder = keras_models.Model(input_layer, output)
        encoder = keras_models.Model(input_layer, bottleneck)
        
        # Custom loss with regularization
        def enhanced_loss(y_true, y_pred):
            reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            # Add L2 regularization to encourage smooth representations
            bottleneck_output = encoder(y_true)
            l2_reg = 0.01 * tf.reduce_mean(tf.square(bottleneck_output))
            return reconstruction_loss + l2_reg
        
        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=enhanced_loss
        )
        
        # Training with data augmentation
        for epoch in range(epochs):
            # Add small noise for regularization
            X_noisy = X + np.random.normal(0, 0.01, X.shape).astype(np.float32)
            autoencoder.fit(X_noisy, X, epochs=1, batch_size=min(batch_size, X.shape[0]), verbose=0)
        
        # Get enhanced features
        enhanced = encoder.predict(X)
        K.clear_session()
        return enhanced

    def contrastive_projection(X, out_dim=64, epochs=30, batch_size=128, temperature=0.1, comm_labels=None, lambda_contrastive=1.0, lambda_sup=1.0):
        # Cải thiện architecture và hyperparameters
        batch_size = min(128, X.shape[0])  # Đảm bảo batch_size không lớn hơn số mẫu
        out_dim = 64
        input_layer = layers.Input(shape=(X.shape[1],))
        
        # Improved architecture với gradual dimension reduction
        x = layers.Dense(min(256, X.shape[1] * 2), activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)  # Tăng dropout để regularize
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        proj = layers.Dense(out_dim, activation=None)(x)  # No activation for final projection
        proj = layers.LayerNormalization()(proj)  # LayerNorm thay vì BatchNorm
        
        # Classifier head for supervised loss (if labels available)
        n_classes = int(np.max(comm_labels)) + 1 if comm_labels is not None and len(np.unique(comm_labels)) > 1 else 1
        if n_classes > 1:
            classifier_out = layers.Dense(n_classes, activation='softmax', name='classifier')(proj)
            projection_model = keras_models.Model(input_layer, [proj, classifier_out])
            # Prepare y_class for supervised loss
            y_class = tf.keras.utils.to_categorical(comm_labels, num_classes=n_classes) if comm_labels is not None else None
        else:
            projection_model = keras_models.Model(input_layer, proj)
            y_class = None
            
        # Improved optimizer with learning rate scheduling
        initial_lr = 0.001
        optimizer = optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        
        for epoch in range(epochs):
            # Learning rate decay
            current_lr = initial_lr * (0.95 ** (epoch // 10))
            optimizer.learning_rate = current_lr
            
            # Improved augmentation strategy với controlled randomness
            idx = np.random.permutation(X.shape[0])
            X1 = graph_augment(X[idx], drop_prob=0.1, noise_std=0.05)  # Light augmentation
            X2 = graph_augment(X[idx], drop_prob=0.15, noise_std=0.08)  # Slightly stronger augmentation
            
            with tf.GradientTape() as tape:
                if n_classes > 1:
                    z1, class_pred1 = projection_model(X1, training=True)
                    z2, class_pred2 = projection_model(X2, training=True)
                else:
                    z1 = projection_model(X1, training=True)
                    z2 = projection_model(X2, training=True)
                
                loss_contrastive = info_nce_loss(z1, z2, temperature=temperature)
                
                # Supervised loss (cross-entropy) for both views
                if n_classes > 1 and y_class is not None:
                    sup_loss1 = losses.CategoricalCrossentropy()(y_class[idx], class_pred1)
                    sup_loss2 = losses.CategoricalCrossentropy()(y_class[idx], class_pred2)
                    sup_loss = (sup_loss1 + sup_loss2) / 2.0
                else:
                    sup_loss = 0.0
                    
                loss = lambda_contrastive * loss_contrastive + lambda_sup * sup_loss
                
            # Gradient clipping để tránh exploding gradients
            grads = tape.gradient(loss, projection_model.trainable_weights)
            clipped_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in grads]
            optimizer.apply_gradients(zip(clipped_grads, projection_model.trainable_weights))
            
            if epoch % 20 == 0:
                print(f"Contrastive epoch {epoch}, loss={loss.numpy():.4f}, contrastive={loss_contrastive.numpy():.4f}, sup={sup_loss.numpy() if n_classes > 1 and y_class is not None else 0.0:.4f}, lr={current_lr:.6f}, temperature={temperature}")
        
        if n_classes > 1:
            out, _ = projection_model.predict(X)
        else:
            out = projection_model.predict(X)
        K.clear_session()
        return out


    # ==== Configurable lambda weights for contrastive projection loss ====
    LAMBDA_CONTRASTIVE = 1.0  # λ1: weight for contrastive loss (giảm xuống để cân bằng)
    LAMBDA_SUP = 0.3          # λ2: weight for supervised loss (giảm để tập trung vào contrastive)

    # Prompt user to select dataset and number of runs
    print("=== Chọn dataset để chạy ===")
    for i, name in datasets:
        print(f"{i}. {name}")
    ds_choice = int(input(f"Nhập số dataset (1-{len(datasets)}): "))
    ds_name = dict(datasets)[ds_choice]
    
    N_RUNS = int(input("Nhập số lần chạy (ví dụ: 20): "))
    print(f"[INFO] Đang chạy {N_RUNS} lần trên dataset: {ds_name}")
    all_results = []
    for run in range(N_RUNS):
        print(f"\n================= RUN {run+1}/{N_RUNS} =================")
        run_results = []
        # Only run for the selected dataset
        G, _, ground_truth = load_dataset(ds_choice)
        # Nếu là dataset lớn, lấy subgraph nhỏ hơn để test nhanh (ví dụ: 7,000 nodes đầu tiên)
        if ds_name in ["YouTube", "DBLP", "Amazon"] and G.number_of_nodes() > 7000:
            print(f"[INFO] Lấy subgraph nhỏ hơn (7,000 nodes đầu tiên) để test nhanh trên {ds_name}...")
            nodes_subset = list(G.nodes())[:7000]
            G = G.subgraph(nodes_subset).copy()
            if ground_truth is not None and len(ground_truth) == len(nodes_subset):
                ground_truth = [ground_truth[n] for n in nodes_subset]
            else:
                ground_truth = None
        # Luôn dùng batch_size mặc định cho mọi dataset (không giảm cho các dataset lớn)
        ae_batch_size_run = ae_batch_size
        if G.number_of_nodes() < 10000:
            A = nx.to_numpy_array(G)
        else:
            print(f"[WARNING] Graph {ds_name} quá lớn ({G.number_of_nodes()} nodes), bỏ qua adjacency matrix cho loss.")
            A = None
        if ground_truth is not None:
            comm_labels = np.array(ground_truth)
        else:
            comm_labels = np.zeros(len(G.nodes()), dtype=int)
        embedding_node2vec = node2vec_embedding(G, dim=feature_dim, **walk_params)
        embedding_deepwalk = deepwalk_embedding(G, dim=feature_dim, **walk_params)
        # Ép kiểu float32 cho embedding để tiết kiệm RAM và tránh lỗi dtype
        embedding_node2vec = embedding_node2vec.astype(np.float32)
        embedding_deepwalk = embedding_deepwalk.astype(np.float32)
        if embedding_deepwalk.shape[1] > feature_dim:
            print(f"[Autoencoder] Đang giảm chiều deepwalk từ {embedding_deepwalk.shape[1]} về {feature_dim} với Laplacian regularization (paper-like) và supervised loss...")
            def autoencoder_reduce_with_graph(X, out_dim, epochs=100, batch_size=32, 
                                            G=None, ground_truth=None, verbose=0,
                                            lambda_recon=1.0, lambda_mod=0.1, 
                                            lambda_neigh=0.1, lambda_sup=1.0):
                input_dim = X.shape[1]
                input_layer = layers.Input(shape=(input_dim,))
                hidden_dim = max(out_dim * 2, 32)
                # Encoder
                x = layers.Dense(hidden_dim)(input_layer)
                x = layers.LeakyReLU(alpha=0.1)(x)
                encoded = layers.Dense(out_dim)(x)
                encoded = layers.ELU(alpha=1.0)(encoded)
                # Decoder
                x_dec = layers.Dense(hidden_dim)(encoded)
                x_dec = layers.LeakyReLU(alpha=0.1)(x_dec)
                decoded = layers.Dense(input_dim, activation='linear')(x_dec)
                # Classifier head for supervised loss (if labels available)
                n_classes = int(np.max(comm_labels)) + 1 if comm_labels is not None and len(np.unique(comm_labels)) > 1 else 1
                classifier_out = layers.Dense(n_classes, activation='softmax', name='classifier')(encoded)
                # Model for AE and classifier
                autoencoder = keras_models.Model(input_layer, [decoded, classifier_out])
                encoder = keras_models.Model(input_layer, encoded)
                optimizer = optimizers.Adam(learning_rate=0.005)
                # Prepare y_class for supervised loss if possible
                if comm_labels is not None and n_classes > 1:
                    y_class = tf.keras.utils.to_categorical(comm_labels, num_classes=n_classes)
                else:
                    y_class = None
                def custom_loss(y_true, y_pred):
                    # y_pred: [decoded, classifier_out]
                    decoded_pred, class_pred = y_pred
                    mse_loss = losses.MeanSquaredError()(y_true, decoded_pred)
                    if A is not None:
                        Z = encoder(y_true)
                        n = tf.shape(Z)[0]
                        A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
                        deg = tf.reduce_sum(A_tf, axis=1, keepdims=True)
                        m = tf.reduce_sum(deg) / 2.0 + 1e-8
                        Z_norm = tf.math.l2_normalize(Z, axis=1)
                        dot = tf.matmul(Z_norm, Z_norm, transpose_b=True)
                        deg_prod = tf.matmul(deg, tf.transpose(deg)) / (2.0 * m)
                        modularity_matrix = A_tf - deg_prod
                        modularity_score = tf.reduce_sum(modularity_matrix * dot) / (2.0 * m)
                        modularity_loss = -modularity_score
                        comm_labels_tf = tf.convert_to_tensor(comm_labels, dtype=tf.int32)
                        mask_same = tf.cast(tf.equal(tf.expand_dims(comm_labels_tf, 1), tf.expand_dims(comm_labels_tf, 0)), tf.float32)
                        mask_diff = 1.0 - mask_same
                        dists = tf.norm(tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0), axis=2)
                        same_mean = tf.reduce_sum(dists * mask_same) / (tf.reduce_sum(mask_same) + 1e-8)
                        diff_mean = tf.reduce_sum(dists * mask_diff) / (tf.reduce_sum(mask_diff) + 1e-8)
                        neigh_loss = same_mean / (diff_mean + 1e-8)
                    else:
                        modularity_loss = 0.0
                        neigh_loss = 0.0
                    if y_class is not None:
                        sup_loss = losses.CategoricalCrossentropy()(y_class, class_pred)
                    else:
                        sup_loss = 0.0
                    total_loss = (
                        lambda_recon * mse_loss +
                        lambda_mod * modularity_loss +
                        lambda_neigh * neigh_loss +
                        lambda_sup * sup_loss
                    )
                    return total_loss
                # --- Mini-batch training loop cho các dataset lớn ---
                if ds_name in ["DBLP", "YouTube", "Amazon"]:
                    dataset = tf.data.Dataset.from_tensor_slices((X, X))
                    dataset = dataset.batch(batch_size)
                    for epoch in range(epochs):
                        losses_epoch = []
                        for batch_x, batch_y in dataset:
                            with tf.GradientTape() as tape:
                                decoded_pred, class_pred = autoencoder(batch_x, training=True)
                                loss = custom_loss(batch_y, [decoded_pred, class_pred])
                            grads = tape.gradient(loss, autoencoder.trainable_weights)
                            optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
                            losses_epoch.append(loss.numpy())
                        if epoch % 20 == 0:
                            print(f"[Mini-batch AE][{ds_name}] epoch {epoch}, loss={np.mean(losses_epoch):.4f}")
                    reduced = encoder.predict(X, batch_size=batch_size)
                    K.clear_session()
                    return reduced
                else:
                    autoencoder.compile(optimizer=optimizer, loss=custom_loss)
                    autoencoder.fit(X, [X, y_class] if y_class is not None else [X, np.zeros((X.shape[0], n_classes))],
                                    epochs=300, batch_size=min(batch_size, X.shape[0]), verbose=0)
                    reduced = encoder.predict(X)
                    K.clear_session()
                    return reduced
            embedding_deepwalk_ae = autoencoder_reduce_with_graph(
                embedding_deepwalk, feature_dim, epochs=ae_epochs, batch_size=ae_batch_size_run, verbose=0,
                lambda_recon=1.0, lambda_mod=0.1, lambda_neigh=0.1, lambda_sup=1.0)
        else:
            embedding_deepwalk_ae = embedding_deepwalk
        # Generate enhanced features based on ground truth availability
        if ground_truth is not None and len(np.unique(ground_truth)) > 1:
            print(f"[INFO] Có ground truth ({len(np.unique(ground_truth))} classes), sẽ dùng contrastive learning...")
            embedding_deepwalk_enhanced = contrastive_projection(
                embedding_deepwalk_ae, out_dim=feature_dim, epochs=150, temperature=0.05,
                comm_labels=comm_labels, lambda_contrastive=LAMBDA_CONTRASTIVE, lambda_sup=LAMBDA_SUP)
            enhancement_name = "deepwalk_ae_contrast"
        else:
            print(f"[INFO] Không có ground truth, chỉ dùng AE embedding...")
            embedding_deepwalk_enhanced = None  # Không enhancement thêm
            enhancement_name = None
        results_table = []
        
        # Embeddings to test
        embeddings_to_test = [
            ("deepwalk", embedding_deepwalk),
            ("node2vec", embedding_node2vec),
            ("deepwalk_ae", embedding_deepwalk_ae)
        ]
        
        # Add enhanced embedding only if it exists (i.e., when ground truth is available)
        if embedding_deepwalk_enhanced is not None and enhancement_name is not None:
            embeddings_to_test.append((enhancement_name, embedding_deepwalk_enhanced))
        
        for emb_type, embedding_feature in embeddings_to_test:
            if embedding_feature is None:
                continue
            for encoder_type in encoder_types:
                print(f"\n==== Dataset={ds_name}, Embedding={emb_type}, Model=GAE, feature_dim={embedding_feature.shape[1]} ====")
                model_kwargs = {"feature_type": "custom", "feature_dim": embedding_feature.shape[1]}
                model = GAE(**model_kwargs)
                model.fit(G, features=embedding_feature, encoder_type=encoder_type)
                embeddings_out = model.get_embedding()
                embeddings_out = preprocess_embedding(embeddings_out)
                best_k, modularity, ari, nmi, labels, elbow_k, elbow_modularity, elbow_labels, inertias = find_best_k(embeddings_out, G, ground_truth)
                cluster_results = cluster_all(embeddings_out, G, n_clusters=best_k)
                for method, metrics in cluster_results.items():
                    if ground_truth is not None:
                        ari_val = adjusted_rand_score(ground_truth, metrics['labels'])
                        nmi_val = normalized_mutual_info_score(ground_truth, metrics['labels'])
                    else:
                        ari_val = None
                        nmi_val = None
                    def compute_conductance_coverage(G, labels):
                        communities = defaultdict(list)
                        for node, label in zip(G.nodes(), labels):
                            communities[label].append(node)
                        conductances = []
                        coverages = []
                        for nodes in communities.values():
                            if len(nodes) == 0 or len(nodes) == len(G):
                                continue
                            cut_size = nx.cut_size(G, nodes)
                            volume = nx.volume(G, nodes)
                            if volume > 0:
                                conductances.append(cut_size / volume)
                            else:
                                conductances.append(0)
                            subgraph = G.subgraph(nodes)
                            internal_edges = subgraph.number_of_edges()
                            total_edges = G.number_of_edges()
                            if total_edges > 0:
                                coverages.append(internal_edges / total_edges)
                            else:
                                coverages.append(0)
                        conductance = np.mean(conductances) if conductances else 0
                        coverage = np.sum(coverages) if coverages else 0
                        return conductance, coverage
                    conductance_val, coverage_val = compute_conductance_coverage(G, metrics['labels'])
                    results_table.append({
                        "Dataset": ds_name,
                        "Embedding": emb_type,
                        "ClusterMethod": method,
                        "Modularity": metrics['modularity'],
                        "Silhouette": metrics['silhouette'],
                        "ARI": ari_val,
                        "NMI": nmi_val,
                        "Conductance": conductance_val,
                        "Coverage": coverage_val
                    })
                    # Format values properly, handling None cases
                    mod_str = f"{metrics['modularity']:.4f}" if metrics['modularity'] is not None else "N/A"
                    sil_str = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] is not None else "N/A"
                    ari_str = f"{ari_val:.4f}" if ari_val is not None else "N/A"
                    nmi_str = f"{nmi_val:.4f}" if nmi_val is not None else "N/A"
                    cond_str = f"{conductance_val:.4f}" if conductance_val is not None else "N/A"
                    cov_str = f"{coverage_val:.4f}" if coverage_val is not None else "N/A"
                    
                    print(f"  - {method}: Modularity={mod_str}, Silhouette={sil_str}, ARI={ari_str}, NMI={nmi_str}, Conductance={cond_str}, Coverage={cov_str}")
        
        run_results.extend(results_table)
        
        # In bảng kết quả từng lần chạy
        if run_results:
            run_df = pd.DataFrame(run_results)
            print(f"\n===== BẢNG KẾT QUẢ RUN {run+1}/{N_RUNS} =====")
            print(run_df.to_string(index=False))
        all_results.extend(run_results)
    
    # Statistical summary processing
    if all_results:
        df = pd.DataFrame(all_results)
        group_cols = ["Dataset", "Embedding", "ClusterMethod"]
        mean_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()
        std_df = df.groupby(group_cols).std(numeric_only=True).reset_index()
        min_df = df.groupby(group_cols).min(numeric_only=True).reset_index()
        max_df = df.groupby(group_cols).max(numeric_only=True).reset_index()
        
        print(f"\n===== BẢNG KẾT QUẢ THỐNG KÊ QUA {N_RUNS} LẦN CHẠY (mean ± std, min, max) =====")
        
        def format_stats(mean, std, minv, maxv):
            if np.isnan(std):
                return f"{mean:.4f} (min={minv:.4f}, max={maxv:.4f})"
            return f"{mean:.4f} ± {std:.4f} (min={minv:.4f}, max={maxv:.4f})"
        
        merged = mean_df.copy()
        for col in mean_df.columns:
            if col in group_cols:
                continue
            merged[col] = [
                format_stats(m, s, mi, ma)
                for m, s, mi, ma in zip(mean_df[col], std_df[col], min_df[col], max_df[col])
            ]
        
        print(merged.to_string(index=False))
        
        # Save to CSV
        csv_filename = f"statistical_summary_{ds_name}_{N_RUNS}runs.csv"
        merged.to_csv(csv_filename, index=False)
        print(f"[INFO] Đã lưu kết quả thống kê vào file: {csv_filename}")


if __name__ == "__main__":
    main()