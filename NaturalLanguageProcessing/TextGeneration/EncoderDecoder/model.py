import tensorflow as tf
from config import *

def build_encoder(vocab_size, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS, depth=ENCODER_DEPTH):
    encoder_inputs = tf.keras.layers.Input(shape=(None,), name='encoder_inputs')
    x = tf.keras.layers.Embedding(
        vocab_size, 
        embedding_dim, 
        mask_zero=True, 
        name='encoder_embedding'
    )(encoder_inputs)
    states_h = []
    states_c = []
    for i in range(depth):
        return_sequences = (i < depth - 1) 
        lstm_layer = tf.keras.layers.LSTM(
            lstm_units, 
            return_sequences=return_sequences, 
            return_state=True, 
            name=f'encoder_lstm_{i+1}'
        )
        if i == 0:
            x, state_h, state_c = lstm_layer(x)
        else:
            x, state_h, state_c = lstm_layer(x)
        states_h.append(state_h)
        states_c.append(state_c)
    encoder_states = states_h + states_c
    encoder_model = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_states, name='encoder')
    return encoder_model, encoder_states

def build_decoder(vocab_size, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS, depth=DECODER_DEPTH):
    decoder_inputs = tf.keras.layers.Input(shape=(None,), name='decoder_inputs')
    decoder_state_inputs = []
    for i in range(depth):
        h = tf.keras.layers.Input(shape=(lstm_units,), name=f'decoder_h_input_{i+1}')
        c = tf.keras.layers.Input(shape=(lstm_units,), name=f'decoder_c_input_{i+1}')
        decoder_state_inputs.extend([h, c])
    state_h = decoder_state_inputs[:depth]
    state_c = decoder_state_inputs[depth:]
    x = tf.keras.layers.Embedding(
        vocab_size, 
        embedding_dim, 
        mask_zero=True, 
        name='decoder_embedding'
    )(decoder_inputs)
    decoder_lstm_outputs = []
    states_h = []
    states_c = []
    
    for i in range(depth):
        lstm_layer = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            return_state=True,
            name=f'decoder_lstm_{i+1}'
        )
        initial_state = [state_h[i], state_c[i]]
        if i == 0:
            x, h, c = lstm_layer(x, initial_state=initial_state)
        else:
            x, h, c = lstm_layer(x, initial_state=initial_state)
        states_h.append(h)
        states_c.append(c)
    decoder_outputs = tf.keras.layers.Dense(
        vocab_size, 
        activation='softmax', 
        name='decoder_output'
    )(x)
    decoder_states = states_h + states_c
    decoder_model = tf.keras.Model(
        inputs=[decoder_inputs] + decoder_state_inputs,
        outputs=[decoder_outputs] + decoder_states,
        name='decoder'
    )
    return decoder_model

def build_seq2seq_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS, 
                        encoder_depth=ENCODER_DEPTH, decoder_depth=DECODER_DEPTH):
    encoder_inputs = tf.keras.layers.Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = tf.keras.layers.Embedding(
        vocab_size, 
        embedding_dim, 
        mask_zero=True, 
        name='encoder_embedding'
    )(encoder_inputs)
    encoder_states_h = []
    encoder_states_c = []
    x = encoder_embedding
    for i in range(encoder_depth):
        return_sequences = (i < encoder_depth - 1)
        encoder_lstm = tf.keras.layers.LSTM(
            lstm_units, 
            return_sequences=return_sequences, 
            return_state=True, 
            name=f'encoder_lstm_{i+1}'
        )
        if i == 0:
            x, state_h, state_c = encoder_lstm(x)
        else:
            x, state_h, state_c = encoder_lstm(x)
        
        encoder_states_h.append(state_h)
        encoder_states_c.append(state_c)
    encoder_states = encoder_states_h + encoder_states_c
    decoder_inputs = tf.keras.layers.Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = tf.keras.layers.Embedding(
        vocab_size, 
        embedding_dim, 
        mask_zero=True, 
        name='decoder_embedding'
    )(decoder_inputs)
    state_h = encoder_states[:encoder_depth]
    state_c = encoder_states[encoder_depth:]
    x = decoder_embedding
    for i in range(decoder_depth):
        decoder_lstm = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            return_state=True,
            name=f'decoder_lstm_{i+1}'
        )
        initial_state = [state_h[i], state_c[i]]
        if i == 0:
            x, _, _ = decoder_lstm(x, initial_state=initial_state)
        else:
            x, _, _ = decoder_lstm(x, initial_state=initial_state)
    decoder_outputs = tf.keras.layers.Dense(
        vocab_size, 
        activation='softmax', 
        name='decoder_output'
    )(x)
    model = tf.keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name='seq2seq'
    )
    
    return model

def create_inference_encoder(model):
    encoder_inputs = model.get_layer('encoder_inputs').input
    encoder_states = []
    for i in range(ENCODER_DEPTH):
        encoder_lstm = model.get_layer(f'encoder_lstm_{i+1}')
        x = model.get_layer('encoder_embedding')(encoder_inputs)
        for j in range(i):
            prev_lstm = model.get_layer(f'encoder_lstm_{j+1}')
            x = prev_lstm(x)[0]  
        _, h, c = encoder_lstm(x)
        encoder_states.extend([h, c])
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
    return encoder_model

def create_inference_decoder(model):
    decoder_inputs = tf.keras.layers.Input(shape=(None,), name='inference_decoder_inputs')
    decoder_states_inputs = []
    for i in range(DECODER_DEPTH):
        h = tf.keras.layers.Input(shape=(LSTM_UNITS,), name=f'inference_decoder_h_input_{i+1}')
        c = tf.keras.layers.Input(shape=(LSTM_UNITS,), name=f'inference_decoder_c_input_{i+1}')
        decoder_states_inputs.extend([h, c])
    decoder_embedding = model.get_layer('decoder_embedding')
    x = decoder_embedding(decoder_inputs)
    decoder_states = []
    for i in range(DECODER_DEPTH):
        decoder_lstm = model.get_layer(f'decoder_lstm_{i+1}')
        layer_states = [decoder_states_inputs[i], decoder_states_inputs[i+DECODER_DEPTH]]
        if i == 0:
            x, state_h, state_c = decoder_lstm(x, initial_state=layer_states)
        else:
            x, state_h, state_c = decoder_lstm(x, initial_state=layer_states)
        decoder_states.extend([state_h, state_c])
    decoder_dense = model.get_layer('decoder_output')
    decoder_outputs = decoder_dense(x)
    decoder_model = tf.keras.Model(
        inputs=[decoder_inputs] + decoder_states_inputs,
        outputs=[decoder_outputs] + decoder_states
    )
    return decoder_model

if __name__ == '__main__':
    model = build_seq2seq_model()
    model.summary()