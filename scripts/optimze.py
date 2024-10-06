import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import rasterio

# Data Generator to handle large datasets efficiently
class TifDataGenerator(Sequence):
    def __init__(self, tif_folder, batch_size=32, sequence_length=10, downscale_factor=4):
        self.tif_files = [os.path.join(tif_folder, file) for file in os.listdir(tif_folder) if file.endswith('.tif')]
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.downscale_factor = downscale_factor
        self.indices = np.arange(len(self.tif_files))

    def __len__(self):
        return int(np.floor(len(self.tif_files) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X, batch_y = self._generate_data(batch_indices)
        return batch_X, batch_y

    def _generate_data(self, batch_indices):
        X = []
        y = []
        for i in batch_indices:
            file_path = self.tif_files[i]
            with rasterio.open(file_path) as src:
                file_data = src.read(1)  # Read as NumPy array
                file_data = file_data[::self.downscale_factor, ::self.downscale_factor]  # Downsample
                X.append(file_data[:-1])  # Take the sequence
                y.append(file_data[1:])   # Take the target for prediction
        return np.array(X), np.array(y)

def create_optimized_model(input_shape, output_dim):
    """
    Create and compile an optimized LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer with dropout
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))  # Increased dropout to reduce overfitting

    # Second LSTM layer
    model.add(LSTM(64, return_sequences=False))  # No return_sequences since it's the last LSTM layer
    model.add(Dropout(0.3))
    
    # Dense layer for output
    model.add(Dense(output_dim))
    
    # Compile the model with a lower learning rate for better convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Lower learning rate
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_optimized_model(tif_folder, sequence_length=10, batch_size=32, epochs=50):
    """
    Train the optimized LSTM model using the TifDataGenerator.
    """
    # Initialize data generators
    train_generator = TifDataGenerator(tif_folder, batch_size=batch_size, sequence_length=sequence_length)

    # Get input and output shapes for the model
    sample_X, sample_y = train_generator.__getitem__(0)
    input_shape = (sample_X.shape[1], sample_X.shape[2])
    output_dim = sample_y.shape[1]

    # Create and compile the optimized model
    model = create_optimized_model(input_shape, output_dim)
    
    # Callbacks for early stopping, saving the best model, and reducing learning rate on plateau
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(f'./models/optimized_model.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save(f'./models/final_optimized_model.keras')
    print(f"Optimized model saved successfully!")

if __name__ == "__main__":
    tif_folder = './data/dataset/'  # Update to the folder containing your .tif files
    train_optimized_model(tif_folder)
