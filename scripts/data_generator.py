import numpy as np
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size=32, sequence_length=10, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data) - sequence_length)
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor((len(self.data) - self.sequence_length) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # Generate data for a batch
        X = np.empty((len(indexes), self.sequence_length, self.data.shape[1]))
        y = np.empty((len(indexes), self.labels.shape[1]))

        for i, idx in enumerate(indexes):
            X[i] = self.data[idx:idx + self.sequence_length]
            y[i] = self.labels[idx + self.sequence_length]

        return X, y
