import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# Load dataset
dataset, info = tfds.load('tiny_shakespeare', with_info=True, as_supervised=False)
text = next(iter(dataset['train']))['text'].numpy().decode('utf-8')

# Create vocabulary
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert text to numerical representation
text_as_int = np.array([char2idx[c] for c in text])

# Define sequence length
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create dataset for training
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    return chunk[:-1], chunk[1:]


dataset = sequences.map(split_input_target)

# Hyperparameters
BATCH_SIZE = 64
BUFFER_SIZE = 10000
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
EPOCHS = 10

# Prepare dataset
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)


# Build model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size, stateful=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=stateful,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)  # Output logits, no activation
    ])
    return model


# Instantiate and compile model
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE, stateful=True)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# Setup checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, ".weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Train model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Load latest checkpoint
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1, stateful=True)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


# Text generation function
def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


# Generate text
print(generate_text(model, start_string=u"Queen: So, let's end this"))
