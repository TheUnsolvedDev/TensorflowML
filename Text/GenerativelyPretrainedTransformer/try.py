import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

# Configuration
vocab_size = 10000  # Vocabulary size
max_seq_length = 50  # Maximum sequence length
embedding_dim = 512  # Embedding dimension
num_heads = 8  # Number of attention heads
num_layers = 4  # Number of transformer layers
dff = 2048  # Dimensionality of the feed-forward network
dropout_rate = 0.1  # Dropout rate

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

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

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
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
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

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

class GPTModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, rate=0.1):
        super(GPTModel, self).__init__()

        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = Embedding(max_seq_len, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = Dense(vocab_size)

    def call(self, x, training=True, mask=None):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_embedding(positions)

        x = self.token_embedding(x) + positions
        x = self.dropout(x, training=training)

        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x, training, mask)

        final_output = self.final_layer(x)

        return final_output

# Dummy data for demonstration
sample_data = np.random.randint(0, vocab_size, (64, max_seq_length))
sample_labels = np.random.randint(0, vocab_size, (64, max_seq_length))

# Create the GPT model
gpt_model = GPTModel(num_layers, embedding_dim, num_heads, dff, vocab_size, max_seq_length, dropout_rate)

# Compile the model
gpt_model.compile(optimizer=Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Train the model
gpt_model.fit(sample_data, sample_labels, epochs=10)
