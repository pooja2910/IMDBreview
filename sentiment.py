import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)

top_words = 8000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)



div = 0.8 * len(X_train)
train = X_train[:int(div)]
val = X_train[int(div):]
train_y = y_train[:int(div)]
val_y =y_train[int(div):]


model = Sequential()
model.add(Embedding(top_words, 64, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))


model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(train, train_y, validation_data=(val, val_y), epochs=100, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test[:5000], y_test[:5000], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

