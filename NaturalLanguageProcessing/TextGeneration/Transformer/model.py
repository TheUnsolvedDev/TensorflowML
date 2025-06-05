import tensorflow as tf
import numpy as np
from config import *

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, d_model):
        super().__init__()
        pos = tf.cast(tf.range(maxlen)[:, None], tf.float32)   # (maxlen,1)
        i   = tf.cast(tf.range(d_model)[None, :], tf.float32)  # (1,d_model)
        angle_rates = 1.0 / tf.pow(
            10000.0,
            (2.0 * tf.math.floor(i/2.0)) / tf.cast(d_model, tf.float32)
        )
        angle_rads = pos * angle_rates                          # (maxlen,d_model)
        # sin on even, cos on odd
        sin = tf.sin(angle_rads[:, 0::2])
        cos = tf.cos(angle_rads[:, 1::2])
        # interleave back into shape (maxlen,d_model)
        angle_rads = tf.reshape(
            tf.stack([sin, cos], axis=-1),
            (maxlen, d_model)
        )
        self.pos_encoding = angle_rads[None, ...]               # (1,maxlen,d_model)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

def transformer_block(x, mask, d_model, num_heads, ff_dim, rate):
    # instantiate MHA
    mha_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=rate
    )
    # call it with keyword args
    attn = mha_layer(
        query=x,
        value=x,
        key=x,
        attention_mask=mask
    )
    attn = tf.keras.layers.Dropout(rate)(attn)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # feed-forward
    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(out1)
    ffn = tf.keras.layers.Dense(d_model)(ffn)
    ffn = tf.keras.layers.Dropout(rate)(ffn)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)

def Transformer_model():
    # 1) Inputs
    input_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")

    # 2) padding mask → (batch,1,1,seq_len)
    mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(tf.not_equal(x, 0), tf.float32)[:, None, None, :]
    )(input_ids)

    # 3) embedding + scale + positional encoding
    x = tf.keras.layers.Embedding(VOCAB_SIZE, D_MODEL, name="embed")(input_ids)
    x = tf.keras.layers.Lambda(
        lambda t: t * tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    )(x)
    x = PositionalEncoding(MAX_LEN, D_MODEL)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)

    # 4) stack Transformer layers
    for _ in range(NUM_LAYERS):
        x = transformer_block(x, mask, D_MODEL, NUM_HEADS, FF_DIM, DROPOUT)

    # 5) final projection
    logits = tf.keras.layers.Dense(VOCAB_SIZE, name="logits")(x)

    return tf.keras.Model(inputs=input_ids, outputs=logits, name="functional_transformer")

# --- test ---
if __name__ == "__main__":
    model = Transformer_model_functional()
    model.summary()