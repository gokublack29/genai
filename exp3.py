import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

text = "catastrophecatalystcatalogue"
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
seq_length = 4

X = []
y = []
for i in range(len(text) - seq_length):
    seq_in = text[i: i + seq_length]
    seq_out = text[i + seq_length]
    X.append([char_to_idx[char] for char in seq_in])
    y.append(char_to_idx[seq_out])

X_encoded = [to_categorical(seq, num_classes = len(chars)) for seq in X]
y_encoded = to_categorical(y, num_classes = len(chars))

X_reshaped = np.array(X_encoded)
y_reshaped = np.array(y_encoded)

model = Sequential([
    SimpleRNN(32, input_shape=(seq_length, len(chars)), activation='tanh'),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_reshaped, y_reshaped, epochs=200, batch_size=1, verbose=0)

def predict_next_char_sequence(seed_seq, num_preds=1):
    assert len(seed_seq) == seq_length, f'Input length must be {seq_length}'
    input_seq = [char_to_idx[char] for char in seed_seq]
    input_seq = to_categorical(input_seq, num_classes=len(chars))
    input_seq = input_seq.reshape(1, seq_length, len(chars))
    predicted_sequence = seed_seq
    for _ in range(num_preds):
        predicted_probs = model.predict(input_seq, verbose=0)
        predicted_idx = np.argmax(predicted_probs)
        next_char = idx_to_char[predicted_idx]
        predicted_sequence += next_char
        next_input = input_seq[0][1:]
        next_input = np.append(next_input, to_categorical(predicted_idx, num_classes = len(chars)).reshape(1, len(chars)), axis = 0)
        input_seq = next_input.reshape(1, seq_length, len(chars))
    return predicted_sequence

print(predict_next_char_sequence('cata', num_preds=5))
print(predict_next_char_sequence('cata', num_preds=10))
