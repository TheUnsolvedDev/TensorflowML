import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *
from attention import *


def positional_encoding(length: int, depth: int):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        if vocab_size is not None:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        if hasattr(self, 'embedding'):
            return self.embedding.compute_mask(*args, **kwargs)
        else:
            return None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if hasattr(self, 'embedding'):
            x = self.embedding(x)
        length = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, x.dtype))
        x = x + \
            tf.cast(self.pos_encoding[tf.newaxis, :length, :], dtype=x.dtype)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1, activation: str = 'relu'):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff, dropout_rate, activation)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout_rate: float = 0.1, activation: str = 'relu', **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate,
                         activation=activation)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1, activation: str = 'relu'):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(
            d_model, dff, dropout_rate, activation=activation)

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout_rate: float = 0.1, activation: str = 'relu', **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation) for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x


def GPTModel(
    input_vocab_size: int = VOCAB_SIZE,
    target_vocab_size: int = VOCAB_SIZE,
    encoder_input_size: int = MAX_LENGTH+1,
    decoder_input_size: int = MAX_LENGTH,
    num_layers: int = NUM_LAYERS,
    d_model: int = 512,
    num_heads: int = NUM_HEADS,
    dff: int = FEED_FORWARD,
    dropout_rate: float = DROPOUT,
) -> tf.keras.Model:
    encoder_input = tf.keras.layers.Input(
        shape=(encoder_input_size,), dtype=tf.int64)
    decoder_input = tf.keras.layers.Input(
        shape=(decoder_input_size,), dtype=tf.int64)

    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                      dff=dff, vocab_size=input_vocab_size, dropout_rate=dropout_rate)(encoder_input)
    decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                      vocab_size=target_vocab_size, dropout_rate=dropout_rate)(decoder_input, encoder)
    output = tf.keras.layers.Dense(target_vocab_size)(decoder)
    return tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=output)


if __name__ == "__main__":
    model = GPTModel()
    model.summary(expand_nested=True)
    
    data = tf.random.uniform((BATCH_SIZE, MAX_LENGTH))
    print(model(data))
