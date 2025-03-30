import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


# Load and prepare the data
data = pd.read_csv("train.txt", sep=';')
data.columns = ['Text', 'Emotions']
data.dropna(inplace=True)

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode the labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# One-Hot encode the labels
num_classes = len(np.unique(labels))
one_hot_labels = keras.utils.to_categorical(labels, num_classes=num_classes)

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(
    padded_sequences, one_hot_labels, test_size=0.2, random_state=42
)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                    output_dim=128,
                    input_length=max_length
                    ))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=num_classes, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

# Input text for prediction
input_text = "She didn't come today because she lost her dog yesterday!"

# Preprocess the input text
input_sequence = tokenizer.texts_to_sequences([input_text])
input_padded_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')

# Make prediction
prediction = model.predict(input_padded_sequence)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
print(predicted_label)
