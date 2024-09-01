import silence_tensorflow.auto
import tensorflow as tf

from config import *


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(input_seq):
    padding_mask = create_padding_mask(input_seq)
    look_ahead_mask = create_look_ahead_mask(tf.shape(input_seq)[1])
    combined_mask = tf.maximum(padding_mask, look_ahead_mask)
    return combined_mask


class GPTModelClass(tf.keras.Model):
    def __init__(self, num_layers=NUM_LAYERS, d_model=EMBEDDING_DIM, num_heads=NUM_HEADS, dff=FEED_FORWARD, vocab_size=VOCAB_SIZE, max_seq_len=MAX_LENGTH, rate=DROPOUT):
        super(GPTModelClass, self).__init__()

        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)

        self.enc_layers = [EncoderLayer(
            d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training=True, mask=create_masks):
        mask = create_masks(x)
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_embedding(positions)

        x = self.token_embedding(x) + positions
        x = self.dropout(x, training=training)

        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x, training, mask)

        final_output = self.final_layer(x)

        return final_output
    
def GPTModel():
    model = GPTModelClass()
    inputs = tf.keras.layers.Input(shape=(MAX_LENGTH,))
    output = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
    

if __name__ == '__main__':
    model = GPTModel()
    model.summary(expand_nested=True)