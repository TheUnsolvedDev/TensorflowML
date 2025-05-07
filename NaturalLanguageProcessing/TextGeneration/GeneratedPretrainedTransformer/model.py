import tensorflow as tf
from config import VOCAB_SIZE, MAX_LEN

VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in locals() else 20000
MAX_LEN = MAX_LEN if 'MAX_LEN' in locals() else 128


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = self.get_positional_encoding(max_len, d_model)

    def get_positional_encoding(self, max_len, d_model):
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class GPTBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, block_id):
        super().__init__(name=f"gpt_block_{block_id}")
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln1_{block_id}")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln2_{block_id}")

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, use_bias=True, dropout=0.0,
            attention_axes=(1,), name=f"mha_{block_id}"
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(embed_dim),
        ], name=f"ffn_{block_id}")

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=True):
        # Causal mask: prevent attending to future tokens
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        attn_output = self.mha(
            query=self.ln1(x), value=self.ln1(x), key=self.ln1(x),
            attention_mask=causal_mask[tf.newaxis, tf.newaxis, :, :]
        )
        x = x + self.dropout(attn_output, training=training)

        ff_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ff_output, training=training)
        return x


def GPT1_Model(vocab_size=VOCAB_SIZE, embedding_dim=768, num_heads=12, ff_dim=3072,
               sequence_len=MAX_LEN, num_layers=7, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=(sequence_len,), dtype=tf.int32, name="input_ids")

    token_embed = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="token_embedding"
    )(inputs)

    x = PositionalEncoding(sequence_len, embedding_dim)(token_embed)

    for i in range(num_layers):
        x = GPTBlock(embedding_dim, num_heads, ff_dim, dropout_rate, block_id=i)(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)

    # Final linear projection for logits
    logits = tf.keras.layers.Dense(vocab_size, name="logits")(x)

    return tf.keras.Model(inputs=inputs, outputs=logits, name="GPT1StyleTransformer")


if __name__ == "__main__":
    model = GPT1_Model()
    model.summary()
