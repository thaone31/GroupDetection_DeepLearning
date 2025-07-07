import numpy as np
import networkx as nx
from models.gae import GAE
# Không cần import Node2Vec từ models, chỉ dùng node2vec_embedding từ feature_utils
from evaluate import find_best_k
from clustering import cluster_all
from models.feature_utils import node_deep_sum

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
        G = nx.read_edgelist('datasets/email-Eu-core.txt', nodetype=int)
        name = "Email"
        ground_truth = None
    elif choice == 5:
        G = nx.read_edgelist('datasets/facebook_combined.txt', nodetype=int)
        name = "Facebook"
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
        (5, "Facebook")
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


    from models.gat import gat_embedding as gt_embedding

    if mode == 1:
        print("=== Chọn dataset ===")
        for i, name in datasets:
            print(f"{i}. {name}")
        ds_choice = int(input(f"Nhập số (1-{len(datasets)}): "))

        feature_dim = 64

        print("\n=== Chọn model ===")
        for i, name in models:
            print(f"{i}. {name}")
        model_choice = int(input("Nhập số (1/2/3): "))

        print("\n=== Chọn phương pháp clustering ===")
        for i, name in clustering_methods:
            print(f"{i}. {name}")
        print("5. So sánh tất cả phương pháp trên")
        cluster_choice = int(input("Nhập số (1/2/3/5): "))

        G, ds_name, ground_truth = load_dataset(ds_choice)
        from models.feature_utils import deepwalk_embedding, node2vec_embedding, deepwalk_enhanced_embedding
        from models.graph_transformer import graph_transformer_embedding as gt_embedding
        from tensorflow.keras import layers, models, optimizers, losses, backend as K

        def autoencoder_reduce(X, out_dim, epochs=100, batch_size=32, verbose=0):
            input_dim = X.shape[1]
            # Autoencoder cơ bản nhất: 1 encoder (Dense, relu), 1 decoder (Dense, linear)
            input_layer = layers.Input(shape=(input_dim,))
            encoded = layers.Dense(out_dim, activation='relu')(input_layer)
            decoded = layers.Dense(input_dim, activation='linear')(encoded)
            autoencoder = models.Model(input_layer, decoded)
            encoder = models.Model(input_layer, encoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(X, X, epochs=epochs, batch_size=min(batch_size, X.shape[0]), verbose=verbose)
            reduced = encoder.predict(X)
            K.clear_session()
            return reduced
        embedding_dict = {}
        # Tối ưu tham số walk cho các dataset lớn, áp dụng cho tất cả nếu muốn nhanh
        # if ds_name in ["Email"]:
        #     walk_params = dict(num_walks=5, walk_length=20)
        # else:

        walk_params = dict()
        embedding_dict["deepwalk"] = deepwalk_embedding(G, dim=feature_dim, **walk_params)
        embedding_dict["node2vec"] = node2vec_embedding(G, dim=feature_dim, **walk_params)
        embedding_dict["graph_transformer"] = gt_embedding(G, dim=feature_dim)
        embedding_dict["node2vec_ae"] = node_deep_sum(G, dim=feature_dim, alpha=0.7, beta=0.3, **walk_params)
        # DeepWalk + MLP nonlinear (BatchNorm, Dropout)
        try:
            print(f"[DeepWalk Enhanced] Đang tạo embedding deepwalk_enhanced (DeepWalk + MLP nonlinear, BatchNorm, Dropout)...")
            embedding_dict["deepwalk_enhanced"] = deepwalk_enhanced_embedding(G, dim=feature_dim, hidden_dim=128, num_layers=2, dropout=0.3, activation='relu', epochs=50, batch_size=32)
        except Exception as e:
            print("Không thể tạo embedding deepwalk_enhanced:", e)
            embedding_dict["deepwalk_enhanced"] = embedding_dict["deepwalk"]
        # node2vec_ae: weighted concat node2vec & deepwalk rồi giảm chiều bằng autoencoder về 64 nếu cần
        try:
            alpha, beta = 0.7, 0.3
            concat = np.concatenate([
                alpha * embedding_dict["node2vec"], beta * embedding_dict["deepwalk"]
            ], axis=1)
            if concat.shape[1] > feature_dim:
                print(f"[Autoencoder] Đang giảm chiều concat node2vec+deepwalk từ {concat.shape[1]} về {feature_dim}...")
                embedding_dict["node2vec_ae"] = autoencoder_reduce(concat, feature_dim, epochs=100, batch_size=32, verbose=0)
            else:
                embedding_dict["node2vec_ae"] = concat
        except Exception as e:
            print("Không thể tạo embedding node2vec_ae:", e)
            embedding_dict["node2vec_ae"] = embedding_dict["node2vec"]
        # Loại bỏ residual, chỉ còn GCN và SAGE
        encoder_types = ["gcn"] if model_choice == 1 else ["sage"]

        print("=== Chọn embedding ===")
        for i, name in embeddings:
            print(f"{i}. {name}")
        emb_choice = int(input(f"Nhập số (1-{len(embeddings)}): "))
        emb_type = dict(embeddings)[emb_choice]
        embedding_feature = embedding_dict[emb_type]

        # Bỏ phase GAE-encoder, trực tiếp cluster từ embedding
        print(f"\nĐang chạy: Dataset={ds_name}, Embedding={emb_type}, Cluster trực tiếp từ embedding, feature_dim={embedding_feature.shape[1]} ...")
        embeddings_out = preprocess_embedding(embedding_feature)
        best_k, modularity, ari, nmi, labels, elbow_k, elbow_modularity, elbow_labels, inertias = find_best_k(embeddings_out, G, ground_truth)
        print(f"\nKết quả tìm số cụm tối ưu cho embedding {emb_type}:")
        print(f" - Số cụm tối ưu: {best_k}")
        if ari is not None and nmi is not None:
            print(f" - ARI: {ari:.4f}")
            print(f" - NMI: {nmi:.4f}")
        else:
            print(" - Không có ground truth để tính ARI/NMI.")
        cluster_results = cluster_all(embeddings_out, G, n_clusters=best_k)
        if cluster_choice == 5:
            print(f"\nSo sánh các thuật toán clustering cho embedding {emb_type}:")
            for method, metrics in cluster_results.items():
                print(f"{method}:")
                print(f"  Modularity: {metrics['modularity']}")
                print(f"  Silhouette: {metrics['silhouette']}")
                print(f"  Số cụm phát hiện: {len(set(metrics['labels']))}")
                print()
        else:
            method_map = {
                1: "KMeans",
                2: "Spectral",
                3: "Agglomerative",
            }
            method = method_map.get(cluster_choice)
            metrics = cluster_results[method]
            print(f"\nKết quả clustering bằng {method} cho embedding {emb_type}:")
            print(f" - Modularity: {metrics['modularity']}")
            print(f" - Silhouette: {metrics['silhouette']}")
            print(f" - Số cụm phát hiện: {len(set(metrics['labels']))}")

    elif mode == 2:
        print("Đang chạy toàn bộ các trường hợp so sánh...")
        feature_dim = 64
        from models.feature_utils import deepwalk_embedding, node2vec_embedding
        from sklearn.decomposition import PCA
        from tensorflow.keras import layers, models, optimizers, losses, backend as K

        def autoencoder_reduce(X, out_dim, epochs=100, batch_size=32, verbose=0):
            input_dim = X.shape[1]
            # Autoencoder đơn giản: 1 encoder, 1 decoder, relu
            input_layer = layers.Input(shape=(input_dim,))
            encoded = layers.Dense(out_dim, activation='relu')(input_layer)
            decoded = layers.Dense(input_dim, activation='linear')(encoded)
            autoencoder = models.Model(input_layer, decoded)
            encoder = models.Model(input_layer, encoded)
            autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.005), loss=losses.MeanSquaredError())
            autoencoder.fit(X, X, epochs=epochs, batch_size=min(batch_size, X.shape[0]), verbose=verbose)
            reduced = encoder.predict(X)
            K.clear_session()
            return reduced
        walk_params = dict()  # Ensure walk_params is always defined
        # Đồng bộ thông số embedding cho tất cả dataset
        walk_params = dict(num_walks=20, walk_length=40)
        feature_dim = 128
        encoder_types = ["gcn", "sage"]
        ae_epochs = 200
        ae_batch_size = 64
        # === Contrastive learning utilities ===
        import tensorflow as tf
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

        def contrastive_projection(X, out_dim=64, epochs=100, batch_size=128, temperature=0.5):
            """
            Contrastive projection with tunable temperature for InfoNCE loss.
            Args:
                X: input embedding
                out_dim: output dim
                epochs: number of epochs
                batch_size: batch size
                temperature: temperature for InfoNCE loss
            """
            from tensorflow.keras import layers, models, optimizers, backend as K
            batch_size = 128
            out_dim = 64
            input_layer = layers.Input(shape=(X.shape[1],))
            proj = layers.Dense(out_dim, activation='relu')(input_layer)
            proj = layers.BatchNormalization()(proj)
            proj = layers.Dropout(0.15)(proj)
            proj = layers.Dense(out_dim, activation=None)(proj)
            proj = layers.BatchNormalization()(proj)
            projection_model = models.Model(input_layer, proj)
            optimizer = optimizers.Adam(learning_rate=0.001)
            for epoch in range(epochs):
                idx = np.random.permutation(X.shape[0])
                X1 = graph_augment(X[idx])
                X2 = graph_augment(X[idx])
                with tf.GradientTape() as tape:
                    z1 = projection_model(X1, training=True)
                    z2 = projection_model(X2, training=True)
                    loss = info_nce_loss(z1, z2, temperature=temperature)
                grads = tape.gradient(loss, projection_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, projection_model.trainable_weights))
                if epoch % 20 == 0:
                    print(f"Contrastive epoch {epoch}, loss={loss.numpy():.4f}, temperature={temperature}")
            out = projection_model.predict(X)
            K.clear_session()
            return out

        import pandas as pd
        for ds_choice, ds_name in datasets:
            G, _, ground_truth = load_dataset(ds_choice)
            embedding_node2vec = node2vec_embedding(G, dim=feature_dim, **walk_params)
            embedding_deepwalk = deepwalk_embedding(G, dim=feature_dim, **walk_params)
            if embedding_deepwalk.shape[1] > feature_dim:
                print(f"[Autoencoder] Đang giảm chiều deepwalk từ {embedding_deepwalk.shape[1]} về {feature_dim}...")
                embedding_deepwalk_ae = autoencoder_reduce(embedding_deepwalk, feature_dim, epochs=ae_epochs, batch_size=ae_batch_size, verbose=0)
            else:
                embedding_deepwalk_ae = embedding_deepwalk

            results_table = []
            # Chỉ giữ lại các embedding cần thiết
            # Tạo embedding deepwalk_ae_contrast
            def contrastive_projection(X, out_dim=64, epochs=100, batch_size=128, temperature=0.5):
                from tensorflow.keras import layers, models, optimizers, backend as K
                input_layer = layers.Input(shape=(X.shape[1],))
                proj = layers.Dense(out_dim, activation='relu')(input_layer)
                proj = layers.BatchNormalization()(proj)
                proj = layers.Dropout(0.15)(proj)
                proj = layers.Dense(out_dim, activation=None)(proj)
                proj = layers.BatchNormalization()(proj)
                projection_model = models.Model(input_layer, proj)
                optimizer = optimizers.Adam(learning_rate=0.001)
                import tensorflow as tf
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
                for epoch in range(50):
                    idx = np.random.permutation(X.shape[0])
                    X1 = graph_augment(X[idx])
                    X2 = graph_augment(X[idx])
                    with tf.GradientTape() as tape:
                        z1 = projection_model(X1, training=True)
                        z2 = projection_model(X2, training=True)
                        loss = info_nce_loss(z1, z2, temperature=0.5)
                    grads = tape.gradient(loss, projection_model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, projection_model.trainable_weights))
                out = projection_model.predict(X)
                K.clear_session()
                return out

            embedding_deepwalk_ae_contrast = contrastive_projection(embedding_deepwalk_ae, out_dim=feature_dim, epochs=50, temperature=0.5)



            for emb_type, embedding_feature in [
                    ("deepwalk", embedding_deepwalk),
                    ("node2vec", embedding_node2vec),
                    ("gat", emb_gat),
                    ("deepwalk_ae", embedding_deepwalk_ae),
                    ("deepwalk_ae_contrast", embedding_deepwalk_ae_contrast)
                ]:
                # Bỏ các embedding fusion nâng cao
                if embedding_feature is None:
                    continue
                if embedding_feature is None:
                    continue
                for encoder_type in encoder_types:
                    print(f"\n==== Dataset={ds_name}, Embedding={emb_type}, Model=GAE, Encoder={encoder_type}, feature_dim={embedding_feature.shape[1]} ====")
                    model_kwargs = {"feature_type": "custom", "feature_dim": embedding_feature.shape[1]}
                    # Chạy 1 lần duy nhất
                    if ds_name == "Email":
                        model = GAE(**model_kwargs)
                        try:
                            model.fit(G, features=embedding_feature, encoder_type=encoder_type)
                        except TypeError as e:
                            print("[WARNING] GAE.fit không nhận epochs/hidden_dim, dùng mặc định.")
                            model.fit(G, features=embedding_feature, encoder_type=encoder_type)
                    else:
                        model = GAE(**model_kwargs)
                        model.fit(G, features=embedding_feature, encoder_type=encoder_type)
                    embeddings_out = model.get_embedding()
                    embeddings_out = preprocess_embedding(embeddings_out)
                    best_k, modularity, ari, nmi, labels, elbow_k, elbow_modularity, elbow_labels, inertias = find_best_k(embeddings_out, G, ground_truth)
                    cluster_results = cluster_all(embeddings_out, G, n_clusters=best_k)
                    for method, metrics in cluster_results.items():
                        # Tính lại ARI, NMI nếu có ground_truth
                        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                        if ground_truth is not None:
                            ari_val = adjusted_rand_score(ground_truth, metrics['labels'])
                            nmi_val = normalized_mutual_info_score(ground_truth, metrics['labels'])
                        else:
                            ari_val = None
                            nmi_val = None
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
                            "NMI": nmi_val
                        })
                        print(f"  - {method}: Modularity={metrics['modularity']}, Silhouette={metrics['silhouette']}, Số cụm={len(set(metrics['labels']))}, ARI={ari_val}, NMI={nmi_val}")
            if results_table:
                df = pd.DataFrame(results_table)
                print("\n===== BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH EMBEDDING, MÔ HÌNH, CLUSTERING TRÊN CÁC DATASET =====")
                print(df.to_string(index=False))
    else:
        print("Chế độ không hợp lệ.")

if __name__ == "__main__":
    main()