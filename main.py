import numpy as np
import networkx as nx
from models.gae import GAE
from evaluate import find_best_k
from clustering import cluster_all
from models.feature_utils import node_deep_sum, deepwalk_embedding, node2vec_embedding, deepwalk_enhanced_embedding
from models.graph_transformer import graph_transformer_embedding as gt_embedding
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, backend as K
import tensorflow.keras.models as keras_models
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict



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
        G = nx.read_edgelist('datasets/com-amazon.ungraph.txt', nodetype=int)
        name = "Amazon"
        ground_truth = None
    elif choice == 8:
        G = nx.read_edgelist('datasets/com-dblp.ungraph.txt', nodetype=int)
        name = "DBLP"
        ground_truth = None
    elif choice == 9:
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
        (5, "Facebook"),
        (6, "Amazon"),
        (7, "DBLP"),
        (8, "YouTube")
    ]
    embeddings = [
        (1, "deepwalk"),
        (2, "node2vec"),
        (3, "node2vec_ae"),
        (4, "deepwalk_enhanced"),
        (5, "graph_transformer"),
    ]
    models = [
        (1, "GAE-GCN (baseline)"),
        (2, "GAE-SAGE (GraphSAGE)")
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
    encoder_types = ["gcn", "sage"]
    ae_epochs = 100  # paper-like: 100 epochs for construction loss
    ae_batch_size = 64
    # Giảm batch_size cho các dataset lớn
    LARGE_DATASETS = ["DBLP", "YouTube", "Amazon"]
    # === Contrastive learning utilities ===
    def graph_augment(X, drop_prob=0.2):
        mask = np.random.binomial(1, 1-drop_prob, X.shape)
        return X * mask

    def info_nce_loss(z1, z2, temperature=0.5):
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        batch_size = tf.shape(z1)[0]
        representations = tf.concat([z1, z2], axis=0)
        similarity_matrix = tf.matmul(representations, representations, transpose_b=True)
        mask = tf.eye(2*batch_size)
        similarity_matrix = similarity_matrix * (1 - mask) - 1e9 * mask
        similarity_matrix /= temperature
        labels = tf.range(batch_size)
        loss1 = tf.keras.losses.sparse_categorical_crossentropy(labels, similarity_matrix[:batch_size, batch_size:], from_logits=True)
        loss2 = tf.keras.losses.sparse_categorical_crossentropy(labels, similarity_matrix[batch_size:, :batch_size], from_logits=True)
        return tf.reduce_mean(loss1 + loss2)

    def contrastive_projection(X, out_dim=64, epochs=30, batch_size=128, temperature=0.005, comm_labels=None, lambda_contrastive=1.0, lambda_sup=1.0):
        batch_size = 128
        out_dim = 64
        input_layer = layers.Input(shape=(X.shape[1],))
        proj = layers.Dense(out_dim, activation='relu')(input_layer)
        proj = layers.BatchNormalization()(proj)
        proj = layers.Dropout(0.15)(proj)
        proj = layers.Dense(out_dim, activation=None)(proj)
        proj = layers.BatchNormalization()(proj)
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
        optimizer = optimizers.Adam(learning_rate=0.001)
        for epoch in range(epochs):
            idx = np.random.permutation(X.shape[0])
            X1 = graph_augment(X[idx])
            X2 = graph_augment(X[idx])
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
            grads = tape.gradient(loss, projection_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, projection_model.trainable_weights))
            if epoch % 20 == 0:
                print(f"Contrastive epoch {epoch}, loss={loss.numpy():.4f}, contrastive={loss_contrastive.numpy():.4f}, sup={sup_loss.numpy() if n_classes > 1 and y_class is not None else 0.0}, temperature={temperature}")
        if n_classes > 1:
            out, _ = projection_model.predict(X)
        else:
            out = projection_model.predict(X)
        K.clear_session()
        return out


    # ==== Configurable lambda weights for contrastive projection loss ====
    LAMBDA_CONTRASTIVE = 2.0  # λ1: weight for contrastive loss
    LAMBDA_SUP = 0.5          # λ2: weight for supervised loss

    # Prompt user to select dataset
    print("=== Chọn dataset để chạy 20 lần ===")
    for i, name in datasets:
        print(f"{i}. {name}")
    ds_choice = int(input(f"Nhập số (1-{len(datasets)}): "))
    ds_name = dict(datasets)[ds_choice]
    print(f"[INFO] Đang chạy 20 lần trên dataset: {ds_name}")

    N_RUNS = 20
    all_results = []
    for run in range(N_RUNS):
        print(f"\n================= RUN {run+1}/{N_RUNS} =================")
        run_results = []
        # Only run for the selected dataset
        G, _, ground_truth = load_dataset(ds_choice)
        # Điều chỉnh batch_size nhỏ cho các dataset lớn
        if ds_name in LARGE_DATASETS:
            ae_batch_size_run = 16
        else:
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
        if embedding_deepwalk.shape[1] > feature_dim:
            print(f"[Autoencoder] Đang giảm chiều deepwalk từ {embedding_deepwalk.shape[1]} về {feature_dim} với Laplacian regularization (paper-like) và supervised loss...")
            def autoencoder_reduce_with_graph(X, out_dim, epochs=100, batch_size=32, verbose=0, laplacian_reg=True, reg_weight=1.0,
                                            lambda_recon=1.0, lambda_mod=0.1, lambda_neigh=0.1, lambda_sup=1.0):
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
                autoencoder.compile(optimizer=optimizer, loss=custom_loss)
                # Fit: y = X, but output is [X, y_class] (y_class only used in loss)
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
        emb_gat = gt_embedding(G, dim=feature_dim)
        embedding_deepwalk_ae_contrast = contrastive_projection(
            embedding_deepwalk_ae, out_dim=feature_dim, epochs=10, temperature=0.005,
            comm_labels=comm_labels, lambda_contrastive=LAMBDA_CONTRASTIVE, lambda_sup=LAMBDA_SUP)
        results_table = []
        for emb_type, embedding_feature in [
                ("deepwalk", embedding_deepwalk),
                ("node2vec", embedding_node2vec),
                ("gat", emb_gat),
                ("deepwalk_ae", embedding_deepwalk_ae),
                ("deepwalk_ae_contrast", embedding_deepwalk_ae_contrast)
            ]:
            if embedding_feature is None:
                continue
            for encoder_type in encoder_types:
                print(f"\n==== Dataset={ds_name}, Embedding={emb_type}, Model=GAE, Encoder={encoder_type}, feature_dim={embedding_feature.shape[1]} ====")
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
                        "Encoder": encoder_type,
                        "ClusterMethod": method,
                        "BestK": best_k,
                        "Modularity": metrics['modularity'],
                        "Silhouette": metrics['silhouette'],
                        "NumClusters": len(set(metrics['labels'])),
                        "ARI": ari_val,
                        "NMI": nmi_val,
                        "Conductance": conductance_val,
                        "Coverage": coverage_val
                    })
                    print(f"  - {method}: Modularity={metrics['modularity']}, Silhouette={metrics['silhouette']}, Số cụm={len(set(metrics['labels']))}, ARI={ari_val}, NMI={nmi_val}, Conductance={conductance_val:.4f}, Coverage={coverage_val:.4f}")
        run_results.extend(results_table)
        # In bảng kết quả từng lần chạy
        if run_results:
            run_df = pd.DataFrame(run_results)
            print(f"\n===== BẢNG KẾT QUẢ RUN {run+1}/{N_RUNS} =====")
            print(run_df.to_string(index=False))
        all_results.extend(run_results)
    if all_results:
        df = pd.DataFrame(all_results)
        group_cols = ["Dataset", "Embedding", "Encoder", "ClusterMethod"]
        mean_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()
        std_df = df.groupby(group_cols).std(numeric_only=True).reset_index()
        min_df = df.groupby(group_cols).min(numeric_only=True).reset_index()
        max_df = df.groupby(group_cols).max(numeric_only=True).reset_index()
        print("\n===== BẢNG KẾT QUẢ THỐNG KÊ QUA 20 LẦN CHẠY (mean ± std, min, max) =====")
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
        # Xuất ra file csv nếu muốn trực quan hóa thêm
        merged.to_csv("statistical_summary.csv", index=False)


if __name__ == "__main__":
    main()
