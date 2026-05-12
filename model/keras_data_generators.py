"""
Data generators for STAIR training.

Expected .npy row format for X:
    row[0] = gt  cell-ID sequence          (T,) or (T,1) int
    row[1] = q   cell-ID sequence          (T,) or (T,1) int
    row[2] = patt_s spatial pattern        (Tp,) float
    row[3] = patt_t temporal pattern       (Tp,) float
    row[4] = gt  time-gap sequence Δt      (T,) float  [optional, zeros if absent]
    row[5] = gt  displacement sequence Δd  (T,) float  [optional, zeros if absent]
    row[6] = q   time-gap sequence Δt      (T,) float  [optional]
    row[7] = q   displacement sequence Δd  (T,) float  [optional]

Expected .npy row format for Y:
    row[0] = gt cell-ID sequence (for reconstruction loss)
    row[1] = spatial pattern target
    row[2] = temporal pattern target
"""

import math
import numpy as np
from tensorflow.keras.utils import Sequence


class TrainGenerator(Sequence):
    def __init__(self, x_path, y_path, batch_size):
        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
        self.batch_size = int(batch_size)

        if len(self.x) != len(self.y):
            raise ValueError(
                f"X and Y length mismatch: {len(self.x)} vs {len(self.y)}"
            )
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_2d(arr, dtype):
        arr = np.asarray(arr, dtype=dtype)
        return arr[:, None] if arr.ndim == 1 else arr

    @staticmethod
    def _safe_row(row, idx, dtype=np.float32, fallback_len=0):
        """Return row[idx] if it exists, else a zero column of length fallback_len."""
        try:
            val = row[idx]
            if val is not None and len(np.asarray(val)) > 0:
                return np.asarray(val, dtype=dtype)
        except (IndexError, TypeError):
            pass
        return np.zeros((fallback_len, 1), dtype=dtype)

    def _pad_to(self, arr_list, target_len, dtype, width=1, fill=0):
        out = np.full((len(arr_list), target_len, width), fill, dtype=dtype)
        for i, arr in enumerate(arr_list):
            arr = self._to_2d(arr, dtype)
            tlen = min(arr.shape[0], target_len)
            wcol = min(arr.shape[1], width)
            out[i, :tlen, :wcol] = arr[:tlen, :wcol]
        return out

    # ── batch construction ────────────────────────────────────────────────────

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x   = self.x[batch_ids]
        batch_y   = self.y[batch_ids]

        gt_ids_l,  q_ids_l,  neg_ids_l  = [], [], []
        gt_dt_l,   q_dt_l,   neg_dt_l   = [], [], []
        gt_dd_l,   q_dd_l,   neg_dd_l   = [], [], []
        patt_s_l,  patt_t_l             = [], []
        y_traj_l,  y_patt_l             = [], []

        n = len(self.x)

        for i, row in enumerate(batch_x):
            gt  = np.asarray(row[0])
            q   = np.asarray(row[1])
            L   = max(len(gt), len(q))

            patt_s = np.asarray(row[2], dtype=np.float32)
            patt_t = np.asarray(row[3], dtype=np.float32)

            # Time gaps and displacements (rows 4-7, optional)
            gt_dt  = self._safe_row(row, 4, np.float32, len(gt))
            gt_dd  = self._safe_row(row, 5, np.float32, len(gt))
            q_dt   = self._safe_row(row, 6, np.float32, len(q))
            q_dd   = self._safe_row(row, 7, np.float32, len(q))

            # Random negative
            neg_idx = batch_ids[i]
            while neg_idx == batch_ids[i]:
                neg_idx = np.random.randint(0, n)
            neg = np.asarray(self.x[neg_idx][0])
            neg_L = len(neg)
            neg_dt = self._safe_row(self.x[neg_idx], 4, np.float32, neg_L)
            neg_dd = self._safe_row(self.x[neg_idx], 5, np.float32, neg_L)

            gt_ids_l.append(gt);   q_ids_l.append(q);   neg_ids_l.append(neg)
            gt_dt_l.append(gt_dt); q_dt_l.append(q_dt); neg_dt_l.append(neg_dt)
            gt_dd_l.append(gt_dd); q_dd_l.append(q_dd); neg_dd_l.append(neg_dd)
            patt_s_l.append(patt_s); patt_t_l.append(patt_t)

            y_traj_l.append(np.asarray(batch_y[i][0]))
            y_ps = np.asarray(batch_y[i][1], dtype=np.float32)
            y_pt = np.asarray(batch_y[i][2], dtype=np.float32)
            y_patt_l.append(np.concatenate(
                [self._to_2d(y_ps, np.float32), self._to_2d(y_pt, np.float32)],
                axis=1,
            ))

        # ── Determine max sequence length ────────────────────────────────────
        all_seqs = gt_ids_l + q_ids_l + neg_ids_l + patt_s_l + patt_t_l
        max_t = max(np.asarray(a).shape[0] for a in all_seqs)

        # ── Pad cell-ID channels  →  main_input (B, 5, max_t, 1) ─────────────
        gt_pad  = self._pad_to(gt_ids_l,  max_t, np.int32)
        q_pad   = self._pad_to(q_ids_l,   max_t, np.int32)
        neg_pad = self._pad_to(neg_ids_l, max_t, np.int32)
        ps_pad  = self._pad_to(patt_s_l,  max_t, np.float32)
        pt_pad  = self._pad_to(patt_t_l,  max_t, np.float32)
        inp_ids = np.stack([gt_pad, q_pad, neg_pad, ps_pad, pt_pad], axis=1)

        # ── Pad Δt channels  →  input_delta_t (B, 3, max_t, 1) ──────────────
        gt_dt_pad  = self._pad_to(gt_dt_l,  max_t, np.float32)
        q_dt_pad   = self._pad_to(q_dt_l,   max_t, np.float32)
        neg_dt_pad = self._pad_to(neg_dt_l, max_t, np.float32)
        inp_dt = np.stack([gt_dt_pad, q_dt_pad, neg_dt_pad], axis=1)

        # ── Pad Δd channels  →  input_delta_d (B, 3, max_t, 1) ──────────────
        gt_dd_pad  = self._pad_to(gt_dd_l,  max_t, np.float32)
        q_dd_pad   = self._pad_to(q_dd_l,   max_t, np.float32)
        neg_dd_pad = self._pad_to(neg_dd_l, max_t, np.float32)
        inp_dd = np.stack([gt_dd_pad, q_dd_pad, neg_dd_pad], axis=1)

        # ── Targets ───────────────────────────────────────────────────────────
        y_traj_pad = self._pad_to(y_traj_l, max_t, np.float32)
        y_patt_pad = self._pad_to(y_patt_l, max_t, np.float32, width=2)
        # Dummy repr target – only shape matters for TripletLoss
        y_repr = np.zeros((len(batch_ids), 3, 256), dtype=np.float32)

        return (
            [inp_ids, inp_dt, inp_dd],
            [y_repr, y_traj_pad, y_patt_pad],
        )
