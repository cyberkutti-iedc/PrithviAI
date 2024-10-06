import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_preprocessed_data(file_path):
    """
    Load preprocessed .npy data file for prediction.
    
    Parameters:
        file_path (str): Path to the .npy file.
        
    Returns:
        np.array: Loaded data.
    """
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"{file_path} not found.")

def create_sequences(data, sequence_length):
    """
    Create sequences of data for LSTM prediction.
    
    Parameters:
        data (np.array): Input data (time_steps, features).
        sequence_length (int): The length of each sequence.
        
    Returns:
        np.array: Sequences (X).
    """
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
    return np.array(X)

def predict_for_dataset(model_name, dataset_name, sequence_length=5):
    """
    Load the trained model, create sequences, and make predictions.
    
    Parameters:
        model_name (str): Name of the trained model (.keras file).
        dataset_name (str): Name of the dataset (.npy file).
        sequence_length (int): The length of sequences for LSTM input.
    """
    try:
        # Load the model
        model_path = f'./models/{model_name}.keras'
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load preprocessed data
        data_path = f'./output/{dataset_name}.npy'
        data = load_preprocessed_data(data_path)
        
        # Flatten spatial dimensions
        data_flattened = data.reshape(data.shape[0], -1)
        
        # Create sequences for prediction
        X_data = create_sequences(data_flattened, sequence_length)
        
        if X_data.shape[0] == 0 or X_data.ndim != 3:
            print(f"Skipping prediction for {dataset_name}: Invalid sequence data.")
            return
        
        # Make predictions
        predictions = model.predict(X_data)
        print(f"Predictions for {dataset_name}:")
        print(predictions)

        # Save predictions
        np.save(f'./output/{dataset_name}_predictions.npy', predictions)
        print(f"Predictions for {dataset_name} saved successfully!")

    except Exception as e:
        print(f"Error while predicting for {dataset_name}: {e}")

def predict_all_models():
    """
    Predict using all models for all datasets.
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
    
    # Predict for each dataset and corresponding model
    for dataset_name in datasets:
        print(f"Predicting for {dataset_name}...")
        predict_for_dataset(f'{dataset_name}_prediction_model', dataset_name)

if __name__ == "__main__":
    predict_all_models()
