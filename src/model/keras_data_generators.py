import math
import numpy as np
from tensorflow.keras.utils import Sequence


class TrainGenerator(Sequence):
    def __init__(self, x_path, y_path, batch_size):
        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
        self.batch_size = int(batch_size)

        if len(self.x) != len(self.y):
            raise ValueError(f"X and Y length mismatch: {len(self.x)} vs {len(self.y)}")

        self.indices = np.arange(len(self.x))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def _to_2d(self, arr, dtype):
        arr = np.asarray(arr, dtype=dtype)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _pad_to(self, arr_list, target_len, dtype, width=1, fill_value=0):
        out = np.full((len(arr_list), target_len, width), fill_value, dtype=dtype)
        for i, arr in enumerate(arr_list):
            arr = self._to_2d(arr, dtype)
            use_len = min(arr.shape[0], target_len)
            use_w = min(arr.shape[1], width)
            out[i, :use_len, :use_w] = arr[:use_len, :use_w]
        return out

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_ids]
        batch_y = self.y[batch_ids]

        gt_list, q_list, neg_list = [], [], []
        patt_s_list, patt_t_list = [], []

        y_traj_list = []
        y_patt_list = []

        n = len(self.x)

        for i, row in enumerate(batch_x):
            gt = np.asarray(row[0])
            q = np.asarray(row[1])
            patt_s = np.asarray(row[2], dtype=np.float32)
            patt_t = np.asarray(row[3], dtype=np.float32)

            neg_idx = np.random.randint(0, n)
            while neg_idx == batch_ids[i]:
                neg_idx = np.random.randint(0, n)
            neg = np.asarray(self.x[neg_idx][0])

            gt_list.append(gt)
            q_list.append(q)
            neg_list.append(neg)
            patt_s_list.append(patt_s)
            patt_t_list.append(patt_t)

            # y = [gt_ids, spatial_pattern, temporal_pattern]
            y_traj = np.asarray(batch_y[i][0])
            y_ps = np.asarray(batch_y[i][1], dtype=np.float32)
            y_pt = np.asarray(batch_y[i][2], dtype=np.float32)

            y_traj_list.append(y_traj)
            y_patt_list.append(np.concatenate([self._to_2d(y_ps, np.float32),
                                               self._to_2d(y_pt, np.float32)], axis=1))

        # max_t is the one used by model outputs
        max_t = max(
            max(np.asarray(a).shape[0] for a in gt_list),
            max(np.asarray(a).shape[0] for a in q_list),
            max(np.asarray(a).shape[0] for a in neg_list),
            max(np.asarray(a).shape[0] for a in patt_s_list),
            max(np.asarray(a).shape[0] for a in patt_t_list),
        )

        gt_pad = self._pad_to(gt_list, max_t, np.int32, width=1, fill_value=0)
        q_pad = self._pad_to(q_list, max_t, np.int32, width=1, fill_value=0)
        neg_pad = self._pad_to(neg_list, max_t, np.int32, width=1, fill_value=0)
        patt_s_pad = self._pad_to(patt_s_list, max_t, np.float32, width=1, fill_value=0.0)
        patt_t_pad = self._pad_to(patt_t_list, max_t, np.float32, width=1, fill_value=0.0)

        x_in = np.stack(
            [gt_pad, q_pad, neg_pad, patt_s_pad, patt_t_pad],
            axis=1
        )  # (B, 5, max_t, 1)

        # IMPORTANT: outputs padded to same max_t
        y_traj_pad = self._pad_to(y_traj_list, max_t, np.float32, width=1, fill_value=0.0)
        y_patt_pad = self._pad_to(y_patt_list, max_t, np.float32, width=2, fill_value=0.0)

        # dummy repr target; only shape matters for custom triplet loss
        y_repr = np.zeros((len(batch_ids), 3, 256), dtype=np.float32)

        return x_in, [y_repr, y_traj_pad, y_patt_pad]
