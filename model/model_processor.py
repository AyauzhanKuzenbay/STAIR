import os
import datetime
import numpy as np
import tensorflow as tf


class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, name="triplet_loss"):
        super().__init__(name=name)
        self.margin = float(margin)

    def call(self, y_true, y_pred):
        # y_pred: (B, 3, D) = [q, gt, neg]
        q = y_pred[:, 0, :]
        gt = y_pred[:, 1, :]
        neg = y_pred[:, 2, :]

        d_pos = tf.reduce_sum(tf.square(q - gt), axis=-1)
        d_neg = tf.reduce_sum(tf.square(q - neg), axis=-1)

        loss = tf.maximum(0.0, d_pos - d_neg + self.margin)
        return tf.reduce_mean(loss)


class ModelProcessor:
    def __init__(self):
        pass

    def model_train(self, model, train_gen, val_gen, arg_processor):
        model_dir = os.path.dirname(arg_processor.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=[
                TripletLoss(arg_processor.triplet_margin),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
            ],
            # IMPORTANT:
            # stack_repr = main retrieval loss
            # out_traj    = disabled (current target is not semantically aligned)
            # out_patt    = weak auxiliary supervision
            loss_weights=[1.0, 0.0, 0.1],
        )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=arg_processor.model_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=arg_processor.patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        print("Training model...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=arg_processor.epochs,
            callbacks=callbacks,
            verbose=1,
        )

    def _pad_id_sequences(self, seq_list):
        max_len = max(len(s) for s in seq_list)
        out = np.zeros((len(seq_list), max_len), dtype=np.int32)

        for i, seq in enumerate(seq_list):
            arr = np.asarray(seq)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            elif arr.ndim > 1:
                arr = arr.reshape(arr.shape[0], -1)[:, 0]
            arr = arr.astype(np.int32)
            out[i, :len(arr)] = arr
        return out

    def _extract_eval_arrays(self, data):
        """
        Expected current preprocess test format:
            row[0] = trajectory id (int)
            row[1] = trajectory ids sequence, shape (T,1)
            row[2] = traj_fourier / extra temporal representation
        We use row[1] for encoder input and row[0] for ground-truth matching.
        """
        ids = []
        trajs = []

        for row in data:
            ids.append(int(row[0]))
            trajs.append(np.asarray(row[1]))

        return np.asarray(ids, dtype=np.int64), trajs

    def _encode_in_batches(self, encoder_model, seqs, batch_size):
        all_vecs = []
        total = len(seqs)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = self._pad_id_sequences(seqs[start:end])

            pred = encoder_model.predict(batch, verbose=0)

            # pred shape: (B, T, H)
            if pred.ndim != 3:
                raise ValueError(f"Expected encoder output rank 3, got shape {pred.shape}")

            # mean pooling over time
            vec = pred.mean(axis=1)
            all_vecs.append(vec)

            if total > batch_size:
                print(f"  batch predict {start}-{end}")

        return np.concatenate(all_vecs, axis=0)

    def _compute_rank_metrics(self, q_ids, q_vecs, db_ids, db_vecs, ks):
        """
        Rank is computed by Euclidean distance in embedding space.
        Ground truth match is db item with the same trajectory id as query id.
        """
        id_to_db_indices = {}
        for i, dbid in enumerate(db_ids):
            id_to_db_indices.setdefault(int(dbid), []).append(i)

        ranks = []
        hits = {k: 0 for k in ks}

        # chunked distance computation to save memory a bit
        db_sq = np.sum(db_vecs * db_vecs, axis=1, keepdims=True).T  # (1, Ndb)

        for i in range(len(q_vecs)):
            qv = q_vecs[i:i+1]  # (1, D)
            qid = int(q_ids[i])

            if qid not in id_to_db_indices:
                # no exact gt in db; skip safely with worst-case rank
                ranks.append(len(db_ids))
                continue

            q_sq = np.sum(qv * qv, axis=1, keepdims=True)           # (1,1)
            cross = qv @ db_vecs.T                                  # (1,Ndb)
            dists = q_sq + db_sq - 2.0 * cross                      # (1,Ndb)
            dists = dists[0]

            order = np.argsort(dists)

            gt_candidates = id_to_db_indices[qid]
            # if multiple same ids exist, take best rank among them
            best_rank = len(db_ids)
            ordered_pos = {idx: pos for pos, idx in enumerate(order)}
            for gt_idx in gt_candidates:
                rank = ordered_pos[gt_idx] + 1  # 1-based rank
                if rank < best_rank:
                    best_rank = rank

            ranks.append(best_rank)

            for k in ks:
                if best_rank <= k:
                    hits[k] += 1

        ranks = np.asarray(ranks, dtype=np.int64)
        topk = [hits[k] / len(q_ids) for k in ks]
        mean_rank = float(np.mean(ranks))
        return ranks, topk, mean_rank

    def model_evaluate(self, encoder_model, arg_processor):
        print("Loading test data...")
        gt_data = np.load(arg_processor.test_gt_path, allow_pickle=True)
        q_data = np.load(arg_processor.test_q_path, allow_pickle=True)

        if os.path.exists(arg_processor.model_path):
            print("Loading best weights...")
            # encoder_model belongs to the bigger model graph, so weights are already
            # loaded into parent model during training via restore_best_weights.
            # If running eval-only later, this line is still helpful when called
            # from a freshly built full model path elsewhere.
            try:
                encoder_model.load_weights(arg_processor.model_path)
            except Exception:
                # encoder-only model cannot directly load whole-model weights.
                # In the train->eval same process, weights are already present.
                print("Skipping direct encoder-only weight load; using current in-memory weights.")
        else:
            print("Warning: model weights file not found. Using current in-memory weights.")

        print("Processing query and database trajectories...")
        gt_ids, gt_trajs = self._extract_eval_arrays(gt_data)
        q_ids, q_trajs = self._extract_eval_arrays(q_data)

        predict_batch_size = getattr(arg_processor, "predict_batch_size", 128)
        ks = list(getattr(arg_processor, "ks", [1, 5, 10, 50]))

        print("Encoding database trajectories...")
        start_time = datetime.datetime.now()
        gt_vecs = self._encode_in_batches(encoder_model, gt_trajs, predict_batch_size)

        print("Encoding query trajectories...")
        q_vecs = self._encode_in_batches(encoder_model, q_trajs, predict_batch_size)

        print("Computing ranks...")
        ranks, topk, mean_rank = self._compute_rank_metrics(
            q_ids=q_ids,
            q_vecs=q_vecs,
            db_ids=gt_ids,
            db_vecs=gt_vecs,
            ks=ks,
        )

        finish_time = datetime.datetime.now()
        total_pred_time = (finish_time - start_time).total_seconds()
        avg_eval_time = total_pred_time / max(len(q_ids), 1)

        print("Finish datetime:", finish_time)
        print("Test GT Shape:", gt_data.shape)
        print("Test Q Shape:", q_data.shape)
        print("Total prediction time:", total_pred_time)
        print("All k:", ks)
        print("Top-k results:", np.array(topk))
        print("Mean rank:", mean_rank)
        print("Average evaluation time:", avg_eval_time)

        output_dir = getattr(arg_processor, "output_directory", "MODEL_OUT")
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "log_eval.txt")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Finish datetime: {finish_time}\n")
            f.write(f"Test GT Shape: {gt_data.shape}\n")
            f.write(f"Test Q Shape: {q_data.shape}\n")
            f.write(f"Total prediction time: {total_pred_time}\n")
            f.write(f"All k: {ks}\n")
            f.write(f"Top-k results: {np.array(topk)}\n")
            f.write(f"Mean rank: {mean_rank}\n")
            f.write(f"Average evaluation time: {avg_eval_time}\n")

        return {
            "ranks": ranks,
            "topk": topk,
            "mean_rank": mean_rank,
            "log_path": log_path,
        }
