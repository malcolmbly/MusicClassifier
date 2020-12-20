# MusicClassifier
In this project, I will attempt to classify the well-known GTZAN music dataset. I want to compare several machine learning methods to improve my understanding of them as well as see how each performs. Methods I plan to include are KNN, Random Forests, and CNN.

#Notes
model = keras.Sequential([
    layers.LSTM(100, input_shape = (timesteps, features), return_sequences=False, stateful=False, dropout = 0.1),
    layers.Dense(100),
    layers.Dense(50, activation = 'tanh'),
    layers.Dense(50),
    layers.Dense(10, activation = 'softmax')
])
gave 92% accuracy over 30 epochs and took about 17 seconds per epoch

model = keras.Sequential([
    layers.Conv1D(filters = 64, kernel_size = 5, input_shape = (timesteps, features)),
    layers.MaxPooling1D(pool_size = 3),
    layers.LSTM(64, return_sequences=False, stateful=False, dropout = 0.1),
    layers.Dense(64),
    layers.Dense(32, activation = 'tanh'),
    layers.Dense(32),
    layers.Dense(10, activation = 'softmax')
])
gave 90% accuracy over 40 epochs and took about 5 seconds per epoch