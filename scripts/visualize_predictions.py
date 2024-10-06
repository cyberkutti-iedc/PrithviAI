import numpy as np
import matplotlib.pyplot as plt
import os

def load_predictions(dataset_name):
    """
    Load predictions from a saved .npy file.
    
    Parameters:
        dataset_name (str): Name of the dataset (used for loading predictions).
        
    Returns:
        np.array: Loaded predictions.
    """
    file_path = f'./output/{dataset_name}_predictions.npy'
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"{file_path} not found.")

def plot_predictions(dataset_name, predictions):
    """
    Plot predictions for a dataset using Matplotlib.
    
    Parameters:
        dataset_name (str): Name of the dataset (for labeling the plot).
        predictions (np.array): Predictions to plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label=f'{dataset_name} Predictions')
    plt.title(f'Predictions for {dataset_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Predicted Value')
    
    # Manually set the legend location to avoid the 'best' location warning
    plt.legend(loc='upper right')  # You can change to other locations like 'lower left', 'center', etc.
    
    plt.show()

if __name__ == "__main__":
    # Example: Load and plot predictions for co2_data
    dataset_name = 'co2_data'
    predictions = load_predictions(dataset_name)
    plot_predictions(dataset_name, predictions)
