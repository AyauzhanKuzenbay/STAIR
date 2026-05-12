"""
STAIR – Spatial-Temporal Similar Trajectory Search under Irregular Time Intervals.

Architecture (paper Sec. 4):
  (i)  IrregularGapEncoder      – hybrid log-Fourier encoding of Δt  (Eq. 1)
  (ii) MultiScaleTemporalFusion – fine / mid / coarse GRU branches    (Sec. 4.2)
  (iii)AttentionPooling         – learned query-vector pooling         (Eq. 14)

Training loss:  L = L_retr  +  λ1·L_p2p  +  λ2·L_patt               (Eq. 25)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Embedding, GRU, Bidirectional,
    Input, Lambda, Concatenate, TimeDistributed,
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


# ---------------------------------------------------------------------------
# (i)  Irregular Gap Encoder  – Eq. (1)
# ---------------------------------------------------------------------------

class IrregularGapEncoder(Layer):
    """
    Maps each inter-point time gap Δt_i to a (1 + 2K)-dim feature vector:

        e_i^(t) = [ log(1 + Δt_i),
                    sin(ω_1 Δt_i), cos(ω_1 Δt_i),
                    ...
                    sin(ω_K Δt_i), cos(ω_K Δt_i) ]

    ω_1 … ω_K are learnable frequencies initialised from N(0, σ^{-2}).
    The dot product e_i^(t) · e_j^(t) approximates a Gaussian kernel over
    gap differences, providing an inductive bias of temporal proximity.
    """

    def __init__(self, K=8, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.K = int(K)
        self.sigma = float(sigma)

    def build(self, input_shape):
        self.freq = self.add_weight(
            name="gap_freq",
            shape=(self.K,),
            initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=1.0 / self.sigma
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, delta_t):
        """
        Args:
            delta_t: (B, T, 1) – time gaps (seconds or minutes)
        Returns:
            (B, T, 1 + 2K)
        """
        log_term = tf.math.log1p(delta_t)                         # (B, T, 1)
        angles   = delta_t * self.freq[None, None, :]             # (B, T, K)
        return tf.concat([log_term, tf.sin(angles), tf.cos(angles)], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"K": self.K, "sigma": self.sigma})
        return cfg


# ---------------------------------------------------------------------------
# Spatial Displacement Encoder  – Eq. (3)
# ---------------------------------------------------------------------------

class SpatialDisplacementEncoder(Layer):
    """Projects scalar Δd_i (cell-centroid Euclidean distance) to d_s-dim."""

    def __init__(self, d_s=16, **kwargs):
        super().__init__(**kwargs)
        self.d_s = int(d_s)
        self.proj = Dense(d_s, use_bias=True, name="disp_proj")

    def call(self, delta_d):
        """
        Args:
            delta_d: (B, T, 1)
        Returns:
            (B, T, d_s)
        """
        return self.proj(delta_d)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_s": self.d_s})
        return cfg


# ---------------------------------------------------------------------------
# (ii) Multi-Scale Temporal Fusion  – Sec. 4.2
# ---------------------------------------------------------------------------

class MultiScaleTemporalFusion(Layer):
    """
    Three-branch GRU encoder at fine / mid / coarse temporal resolutions.

    Mid and coarse branches compress the sequence via average pooling with
    strides s_m and s_c, encode with a dedicated GRU, then upsample back
    to L via nearest-neighbour interpolation.  Branch outputs are fused
    with learnable softmax weights α_f, α_m, α_c  (Eq. 12-13).

    When trajectories are densely sampled the fine branch dominates; when
    sparse and irregular, the coarse branch compensates with broader context.
    """

    def __init__(self, dh=256, s_m=2, s_c=4,
                 num_layers=1, dropout=0.3, bidir=True, **kwargs):
        super().__init__(**kwargs)
        self.dh         = int(dh)
        self.s_m        = int(s_m)
        self.s_c        = int(s_c)
        self.num_layers = int(num_layers)
        self.dropout    = float(dropout)
        self.bidir      = bool(bidir)

    def build(self, input_shape):
        def make_stack(prefix):
            layers = []
            for i in range(self.num_layers):
                gru = GRU(self.dh, return_sequences=True,
                           dropout=self.dropout, name=f"{prefix}_gru_{i}")
                if self.bidir:
                    gru = Bidirectional(gru, name=f"{prefix}_bigru_{i}")
                layers.append(gru)
            return layers

        self.fine_grus   = make_stack("fine")
        self.mid_grus    = make_stack("mid")
        self.coarse_grus = make_stack("coarse")

        # Raw logits → softmax → [α_f, α_m, α_c]  (Eq. 12)
        self.fusion_logits = self.add_weight(
            name="fusion_logits", shape=(3,),
            initializer="zeros", trainable=True,
        )

        out_dim = self.dh * 2 if self.bidir else self.dh
        self.out_proj = Dense(out_dim, activation="relu", name="ms_out_proj")
        super().build(input_shape)

    def _run_stack(self, x, grus, mask=None):
        for gru in grus:
            x = gru(x, mask=mask) if mask is not None else gru(x)
        return x

    def call(self, x, mask=None):
        """
        Args:
            x:    (B, L, E) – concatenated token sequence
            mask: (B, L)    – boolean padding mask
        Returns:
            (B, L, out_dim) – fused hidden states H̃^(n)
        """
        L = tf.shape(x)[1]

        # Fine branch – Eq. (7)
        H_f = self._run_stack(x, self.fine_grus, mask)

        # Mid branch – Eq. (8-9): pool → GRU → upsample
        x_m = tf.nn.avg_pool1d(x, ksize=self.s_m, strides=self.s_m,
                                padding="SAME")
        H_m_small = self._run_stack(x_m, self.mid_grus)
        H_m = tf.image.resize(
            tf.expand_dims(H_m_small, 2), size=[L, 1], method="nearest",
        )[:, :, 0, :]

        # Coarse branch – Eq. (10-11): pool → GRU → upsample
        x_c = tf.nn.avg_pool1d(x, ksize=self.s_c, strides=self.s_c,
                                padding="SAME")
        H_c_small = self._run_stack(x_c, self.coarse_grus)
        H_c = tf.image.resize(
            tf.expand_dims(H_c_small, 2), size=[L, 1], method="nearest",
        )[:, :, 0, :]

        # Adaptive fusion – Eq. (12-13)
        w = tf.nn.softmax(self.fusion_logits)
        fused = w[0] * H_f + w[1] * H_m + w[2] * H_c

        out = self.out_proj(fused)
        if mask is not None:
            out = out * tf.cast(mask[:, :, None], out.dtype)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(dh=self.dh, s_m=self.s_m, s_c=self.s_c,
                        num_layers=self.num_layers, dropout=self.dropout,
                        bidir=self.bidir))
        return cfg


# ---------------------------------------------------------------------------
# (iii) Attention Pooling  – Eq. (14)
# ---------------------------------------------------------------------------

class AttentionPooling(Layer):
    """
    Condenses H̃ = {h̃_i}_{i=1}^L into a fixed-size embedding z via a
    global learnable query vector q:

        a_i = softmax_i( q^T h̃_i )
        z   = Σ_i a_i · h̃_i
    """

    def __init__(self, dh, **kwargs):
        super().__init__(**kwargs)
        self.dh = int(dh)

    def build(self, input_shape):
        self.query = self.add_weight(
            name="attn_query", shape=(self.dh,),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, H, mask=None):
        """
        Args:
            H:    (B, L, dh)
            mask: (B, L)
        Returns:
            z: (B, dh)
        """
        scores = tf.einsum("bld,d->bl", H, self.query)
        if mask is not None:
            scores += (1.0 - tf.cast(mask, scores.dtype)) * -1e9
        weights = tf.nn.softmax(scores, axis=-1)                  # (B, L)
        return tf.einsum("bl,bld->bd", weights, H)                # (B, dh)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dh": self.dh})
        return cfg


# ---------------------------------------------------------------------------
# STAIR Encoder  (Sec. 4.1–4.3 combined)
# ---------------------------------------------------------------------------

class Encoder:
    """
    Builds the STAIR trajectory encoder.

    Token construction (Eq. 4):
        x_i = v_{c_i} || e_i^(t) || e_i^(d)
                  ↓
        MultiScaleTemporalFusion  →  H̃
                  ↓
        AttentionPooling  →  z  (L2-normalised)

    Inputs (all padded to the same length L):
        ids     : (B, L)    – cell IDs  (int32, 0 = pad)
        delta_t : (B, L, 1) – time gaps (float32)
        delta_d : (B, L, 1) – spatial displacements in metres (float32)
    """

    def __init__(self, vocab, emb_size, gru_size, layers,
                 dropout, K=8, d_s=16, bidir=True):

        ids     = Input((None,),   dtype="int32",   name="traj_ids")
        delta_t = Input((None, 1), dtype="float32", name="traj_delta_t")
        delta_d = Input((None, 1), dtype="float32", name="traj_delta_d")

        # Cell embedding  (Eq. 2)
        v = Embedding(vocab, emb_size, mask_zero=True,
                      name="cell_embedding")(ids)                  # (B, L, de)

        # Temporal gap encoding  (Eq. 1)
        e_t = IrregularGapEncoder(K=K, name="gap_encoder")(delta_t)  # (B, L, 1+2K)

        # Spatial displacement encoding  (Eq. 3)
        e_d = SpatialDisplacementEncoder(d_s=d_s, name="disp_encoder")(delta_d)  # (B, L, ds)

        # Concatenate token  x_i = v_ci || e_i^(t) || e_i^(d)  (Eq. 4)
        x = Concatenate(axis=-1, name="token_concat")([v, e_t, e_d])

        # Padding mask
        mask = Lambda(lambda z: tf.not_equal(z, 0), name="traj_mask")(ids)

        # Multi-Scale Temporal Fusion
        # s_m=2, s_c=4 are fixed constants from the paper (Sec. 4.5)
        H = MultiScaleTemporalFusion(
            dh=gru_size, s_m=2, s_c=4,
            num_layers=layers, dropout=dropout, bidir=bidir,
            name="ms_temporal_fusion",
        )(x, mask=mask)                                            # (B, L, dh*2)

        out_dim = gru_size * 2 if bidir else gru_size

        # Attention pooling  (Eq. 14)
        z = AttentionPooling(dh=out_dim, name="attn_pool")(H, mask=mask)  # (B, dh*)

        # L2 normalisation
        z_norm = Lambda(
            lambda v: tf.math.l2_normalize(v, axis=-1), name="l2_norm"
        )(z)

        self.model   = Model([ids, delta_t, delta_d], [z_norm, H],
                             name="stair_encoder")
        self.out_dim = out_dim


# ---------------------------------------------------------------------------
# Full STAIR Training Model
# ---------------------------------------------------------------------------

class STAIRModel:
    """
    Full STAIR training graph with composite loss (Eq. 25):

        L = L_retr  +  λ1 · L_p2p  +  λ2 · L_patt

    Main inputs:
        inp_ids : (B, 5, L, 1) int32
            ch 0 = gt cell IDs, ch 1 = q cell IDs, ch 2 = neg cell IDs,
            ch 3 = patt_s,      ch 4 = patt_t
        inp_dt  : (B, 3, L, 1) float32  – time gaps for [gt, q, neg]
        inp_dd  : (B, 3, L, 1) float32  – spatial displacements for [gt, q, neg]

    If Δt / Δd are unavailable from preprocessing, pass zero arrays; the
    gap encoder degrades gracefully to positional-only mode.
    """

    def __init__(self,
                 embed_vocab_size,
                 embedding_size,
                 traj_repr_size,
                 gru_cell_size,
                 num_gru_layers,
                 gru_dropout_ratio,
                 bidirectional,
                 use_attention,          # kept for API compatibility
                 k):

        if embed_vocab_size is None:
            embed_vocab_size = 20000

        # s_m=2, s_c=4 are fixed per paper Sec. 4.5 (not tuned hyperparameters)
        encoder = Encoder(
            vocab=embed_vocab_size,
            emb_size=embedding_size,
            gru_size=gru_cell_size,
            layers=num_gru_layers,
            dropout=gru_dropout_ratio,
            bidir=bidirectional,
        )
        self.encoder = encoder
        out_dim = encoder.out_dim

        # ── Inputs ───────────────────────────────────────────────────────────
        inp_ids = Input((5, None, 1), name="main_input")
        inp_dt  = Input((3, None, 1), dtype="float32", name="input_delta_t")
        inp_dd  = Input((3, None, 1), dtype="float32", name="input_delta_d")

        def _int_ch(t, ch):
            return tf.cast(t[:, ch, :, 0], tf.int32)

        def _f32_ch(t, ch):
            return t[:, ch, :, :]    # (B, L, 1)

        gt_ids  = Lambda(lambda z: _int_ch(z, 0), name="split_gt_ids")(inp_ids)
        q_ids   = Lambda(lambda z: _int_ch(z, 1), name="split_q_ids")(inp_ids)
        neg_ids = Lambda(lambda z: _int_ch(z, 2), name="split_neg_ids")(inp_ids)
        patt_s  = Lambda(lambda z: tf.cast(z[:, 3, :, :], tf.float32),
                         name="split_patt_s")(inp_ids)
        patt_t  = Lambda(lambda z: tf.cast(z[:, 4, :, :], tf.float32),
                         name="split_patt_t")(inp_ids)

        gt_dt  = Lambda(lambda z: _f32_ch(z, 0), name="split_gt_dt")(inp_dt)
        q_dt   = Lambda(lambda z: _f32_ch(z, 1), name="split_q_dt")(inp_dt)
        neg_dt = Lambda(lambda z: _f32_ch(z, 2), name="split_neg_dt")(inp_dt)

        gt_dd  = Lambda(lambda z: _f32_ch(z, 0), name="split_gt_dd")(inp_dd)
        q_dd   = Lambda(lambda z: _f32_ch(z, 1), name="split_q_dd")(inp_dd)
        neg_dd = Lambda(lambda z: _f32_ch(z, 2), name="split_neg_dd")(inp_dd)

        # ── Encode with shared encoder ───────────────────────────────────────
        zq, H_q = encoder.model([q_ids,   q_dt,   q_dd])
        zg, _   = encoder.model([gt_ids,  gt_dt,  gt_dd])
        zn, _   = encoder.model([neg_ids, neg_dt, neg_dd])

        # Optional projection
        if traj_repr_size and traj_repr_size != out_dim:
            proj = Dense(traj_repr_size, name="repr_proj")
            zq = proj(zq); zg = proj(zg); zn = proj(zn)

        # ── Triplet output  [zq, zg, zn]  →  (B, 3, repr_dim) ──────────────
        stack = Lambda(
            lambda vecs: K.stack(vecs, axis=1), name="stack_repr"
        )([zq, zg, zn])

        # ── Reconstruction head  L_p2p  (Eq. 20) ────────────────────────────
        out_traj = TimeDistributed(
            Dense(k, activation="relu"), name="out_traj"
        )(H_q)                                                     # (B, L, k)

        # ── Pattern head  L_patt  (Eq. 24) ──────────────────────────────────
        patt = Concatenate(axis=-1, name="concat_patt")([patt_s, patt_t])
        out_patt = TimeDistributed(
            Dense(2, activation="relu"), name="out_patt"
        )(patt)                                                    # (B, L, 2)

        self.model = Model(
            [inp_ids, inp_dt, inp_dd],
            [stack, out_traj, out_patt],
            name="stair_model",
        )
