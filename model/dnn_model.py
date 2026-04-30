import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class TERIEncoding(Layer):
    def __init__(self, emb_size, scale=0.01, **kwargs):
        super().__init__(**kwargs)
        self.emb_size = int(emb_size)
        self.scale = float(scale)
        self.half = max(1, self.emb_size // 2)

    def build(self, input_shape):
        self.freq = self.add_weight(
            shape=(self.half,),
            initializer="glorot_uniform",
            trainable=True,
            name="teri_freq"
        )

        self.proj = Dense(self.emb_size, activation="tanh")

        self.alpha = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(self.scale),
            trainable=True,
            name="teri_scale"
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, E)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        pos = tf.cast(tf.range(T), tf.float32)
        pos = tf.reshape(pos, (1, T, 1))
        pos = tf.tile(pos, [B, 1, 1])

        angles = pos * self.freq[None, None, :]
        sin = tf.sin(angles)
        cos = tf.cos(angles)

        fourier = tf.concat([sin, cos], axis=-1)
        fourier = tf.ensure_shape(fourier, [None, None, 2 * self.half])

        enc = self.proj(fourier)
        return x + self.alpha * enc


class SoftWarpLayer(Layer):
    """
    Differentiable warpformer-inspired layer.
    Instead of hard round/gather, it learns a soft gate per timestep.
    This keeps the 'warp' idea but remains trainable and stable.
    """
    def __init__(self, d_model=256, hidden=128, gate_scale=0.10, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.gate_scale = float(gate_scale)

    def build(self, input_shape):
        self.fc1 = Dense(self.hidden, activation="relu")
        self.fc2 = Dense(1, activation="sigmoid")
        self.mix = Dense(self.d_model, activation="tanh")
        self.beta = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(self.gate_scale),
            trainable=True,
            name="warp_beta"
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (B, T, E)
        gate = self.fc1(x)
        gate = self.fc2(gate)  # (B, T, 1)

        warped = self.mix(x)   # (B, T, E)
        out = x + self.beta * gate * warped

        if mask is not None:
            mask = tf.cast(mask[:, :, None], out.dtype)
            out = out * mask

        return out


class Encoder:
    def __init__(self, vocab, emb_size, gru_size, layers, dropout, bidir=True):
        ids = Input((None,), dtype="int32", name="traj_ids")

        emb = Embedding(vocab, emb_size, mask_zero=True, name="cell_embedding")(ids)

        # Soft TERI residual
        emb = TERIEncoding(emb_size, scale=0.01, name="teri_encoding")(emb)

        mask = Lambda(lambda z: tf.not_equal(z, 0), name="traj_mask")(ids)

        # Differentiable Warpformer-inspired gating
        emb = SoftWarpLayer(d_model=emb_size, hidden=max(64, emb_size // 2), name="warp_layer")(emb, mask=mask)

        x = emb

        for i in range(layers):
            if bidir:
                x = Bidirectional(
                    GRU(
                        gru_size,
                        return_sequences=True,
                        dropout=dropout
                    ),
                    name=f"bigru_{i}"
                )(x, mask=mask)
            else:
                x = GRU(
                    gru_size,
                    return_sequences=True,
                    dropout=dropout,
                    name=f"gru_{i}"
                )(x, mask=mask)

        self.model = Model(ids, x, name="encoder_model")


class STSeqModel:
    def __init__(self,
                 embed_vocab_size,
                 embedding_size,
                 traj_repr_size,
                 gru_cell_size,
                 num_gru_layers,
                 gru_dropout_ratio,
                 bidirectional,
                 use_attention,
                 k):

        inp = Input((5, None, 1), name="main_input")

        gt   = Lambda(lambda z: tf.cast(z[:, 0, :, 0], tf.int32), name="split_gt")(inp)
        q    = Lambda(lambda z: tf.cast(z[:, 1, :, 0], tf.int32), name="split_q")(inp)
        neg  = Lambda(lambda z: tf.cast(z[:, 2, :, 0], tf.int32), name="split_neg")(inp)

        patt_s = Lambda(lambda z: tf.cast(z[:, 3, :, :], tf.float32), name="split_patt_s")(inp)
        patt_t = Lambda(lambda z: tf.cast(z[:, 4, :, :], tf.float32), name="split_patt_t")(inp)

        patt = Concatenate(axis=2, name="concat_patt")([patt_s, patt_t])

        if embed_vocab_size is None:
            embed_vocab_size = 20000

        encoder = Encoder(
            embed_vocab_size,
            embedding_size,
            gru_cell_size,
            num_gru_layers,
            gru_dropout_ratio,
            bidirectional
        )

        self.encoder = encoder

        enc_q = encoder.model(q)
        enc_gt = encoder.model(gt)
        enc_neg = encoder.model(neg)

        pool = Lambda(lambda z: tf.reduce_mean(z, axis=1), name="mean_pool")

        rq = pool(enc_q)
        rg = pool(enc_gt)
        rn = pool(enc_neg)

        if traj_repr_size is not None:
            proj = Dense(traj_repr_size, name="repr_proj")
            rq = proj(rq)
            rg = proj(rg)
            rn = proj(rn)

        stack = Lambda(lambda z: K.stack(z, axis=1), name="stack_repr")([rq, rg, rn])

        out_traj = TimeDistributed(
            Dense(k, activation="relu"),
            name="out_traj"
        )(enc_q)

        out_patt = TimeDistributed(
            Dense(2, activation="relu"),
            name="out_patt"
        )(patt)

        self.model = Model(inp, [stack, out_traj, out_patt], name="stseq_model")
