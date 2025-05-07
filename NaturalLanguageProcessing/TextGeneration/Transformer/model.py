import tensorflow as tf
from config import VOCAB_SIZE, MAX_LEN

# Fallback if config import fails
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


def TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, block_id):
    inputs = tf.keras.Input(shape=(None, embed_dim))
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, name=f"mha_{block_id}"
    )(inputs, inputs, attention_mask=None)

    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ff = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(embed_dim),
    ])
    ff_output = ff(out1)
    ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ff_output)

    return tf.keras.Model(inputs=inputs, outputs=out2, name=f"transformer_block_{block_id}")


def Transformer_model(vocab_size=VOCAB_SIZE, embedding_dim=256, num_heads=4, ff_dim=512,
                      sequence_len=MAX_LEN, num_layers=3, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(sequence_len,), dtype=tf.int32, name="input_ids")

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="embedding"
    )(inputs)

    x = PositionalEncoding(sequence_len, embedding_dim)(x)

    for i in range(num_layers):
        x = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout_rate, block_id=i + 1)(x)

    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', name="output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="DeepTextGenTransformer")


if __name__ == '__main__':
    model = Transformer_model()
    model.summary()
