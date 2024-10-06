import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_preprocessed_data(file_path):
    """
    Load preprocessed .npy data file.
    
    Parameters:
        file_path (str): Path to the .npy file.
        
    Returns:
        np.array: Loaded data.
    """
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"{file_path} not found.")

def create_model(input_shape, output_dim):
    """
    Create and compile an LSTM model.
    
    Parameters:
        input_shape (tuple): Shape of the input data (sequence_length, features).
        output_dim (int): Number of output features (should match the shape of the target data).
        
    Returns:
        model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, sequence_length):
    """
    Create sequences of data for LSTM training.
    
    Parameters:
        data (np.array): Input data (time_steps, features).
        sequence_length (int): The length of each sequence.
        
    Returns:
        tuple: Sequences (X) and corresponding targets (y).
    """
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def process_and_train(dataset_name, sequence_length=5):
    """
    Load data, create sequences, train the LSTM model, and save the model.
    
    Parameters:
        dataset_name (str): Name of the dataset (used to load the .npy file and save the model).
        sequence_length (int): Length of the sequences for LSTM training.
    """
    try:
        # Load the preprocessed data
        file_path = f'./output/{dataset_name}.npy'
        data = load_preprocessed_data(file_path)
        
        # Flatten spatial dimensions
        data_flattened = data.reshape(data.shape[0], -1)
        
        # Check if the dataset has enough samples for the given sequence length
        if len(data_flattened) <= sequence_length:
            print(f"Skipping {dataset_name}: not enough data for the sequence length of {sequence_length}.")
            return
        
        # Create sequences
        X_data, y_data = create_sequences(data_flattened, sequence_length)
        
        # Check if sequences are created correctly
        if X_data.shape[0] == 0 or X_data.ndim != 3 or y_data.ndim != 2:
            print(f"Skipping {dataset_name}: Unable to create valid sequences.")
            return
        
        # Define input shape for the model
        input_shape = (X_data.shape[1], X_data.shape[2])  # (sequence_length, features)
        output_dim = y_data.shape[1]  # Number of features we are predicting
        
        # Create and compile the LSTM model
        model = create_model(input_shape, output_dim)
        
        # Train the model
        model.fit(X_data, y_data, epochs=10, batch_size=32, validation_split=0.2)
        
        # Save the trained model in .h5 format
        model.save(f'./models/{dataset_name}_prediction_model.keras')
        print(f"Model for {dataset_name} saved successfully!")

    except Exception as e:
        print(f"Error while processing {dataset_name}: {e}")

def train_all_models():
    """
    Train models for all datasets.
    """
    datasets = [
        'co2_data',
        'methane_data',
        'micasa_data',
        'oco2_data',
        'odic_data',
        'gosat_data',
        'crop_co2_data',
        'flux_rh_data'
    ]
    
    # Train model for each dataset
    for dataset_name in datasets:
        print(f"Training model for {dataset_name}...")
        process_and_train(dataset_name)

if __name__ == "__main__":
    train_all_models()
