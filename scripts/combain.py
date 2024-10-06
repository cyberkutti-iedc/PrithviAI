import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

def resample_raster(source, target_shape):
    """
    Resample the input raster to match the target shape.
    
    Parameters:
        source (rasterio.DatasetReader): The source raster.
        target_shape (tuple): The target shape to resample to (height, width).
    
    Returns:
        np.array: Resampled data.
    """
    data = source.read(
        1,
        out_shape=(target_shape[0], target_shape[1]),  # Resample to target shape
        resampling=Resampling.bilinear  # Bilinear resampling method
    )
    return data

def combine_tif_files(tif_folder):
    """
    Combine multiple .tif files into one dataset for visualization, resampling as needed.
    
    Parameters:
        tif_folder (str): Path to the folder containing .tif files.
        
    Returns:
        np.array: Combined data from the .tif files.
    """
    combined_data = None
    target_shape = None
    
    for root, dirs, files in os.walk(tif_folder):
        for file in files:
            if file.endswith('.tif'):
                filepath = os.path.join(root, file)
                with rasterio.open(filepath) as src:
                    # Set the target shape based on the first .tif file
                    if target_shape is None:
                        target_shape = src.shape
                    
                    # Resample the current .tif file to the target shape
                    data = resample_raster(src, target_shape)
                    
                    # Combine the data (e.g., by maximum value)
                    if combined_data is None:
                        combined_data = data
                    else:
                        combined_data = np.maximum(combined_data, data)

    return combined_data

# Usage
tif_folder_path = "./data/dataset/"
combined_data = combine_tif_files(tif_folder_path)

# Save the combined data as a .npy file
np.save("combined_data.npy", combined_data)
print("Data combined and saved successfully.")
