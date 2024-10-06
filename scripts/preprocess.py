import rasterio
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_tif_data(tif_folder, downscale_factor=4):
    """
    Load and preprocess .tif files with optional downsampling.
    
    Parameters:
        tif_folder (str): Path to the folder containing .tif files.
        downscale_factor (int): Factor to downsample the data (default is 4).
        
    Returns:
        np.array: Downsampled data as a NumPy array.
    """
    data = []
    for root, dirs, files in os.walk(tif_folder):
        for file in files:
            if file.endswith('.tif'):
                filepath = os.path.join(root, file)
                print(f"Loading file: {filepath}")
                with rasterio.open(filepath) as src:
                    file_data = src.read(1)  # Read the raster file into a numpy array
                    if file_data.size > 0:
                        # Downsample the data
                        downsampled_data = file_data[::downscale_factor, ::downscale_factor]
                        data.append(downsampled_data)
                        print(f"Loaded downsampled data {downsampled_data.shape} from {filepath}")
    return np.array(data)

def normalize_data(data):
    """
    Normalize data using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

def save_data(output_folder, filename, data):
    """
    Save the preprocessed data to a .npy file.
    
    Parameters:
        output_folder (str): Folder to save the output files.
        filename (str): Name of the output file.
        data (np.array): Preprocessed data to save.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    filepath = os.path.join(output_folder, filename)
    np.save(filepath, data)
    print(f"Saved {filename} to {output_folder}")

def preprocess_data():
    """
    Preprocess all the datasets and save them in the output folder.
    """
    output_folder = './output'

    # Folder paths for each dataset
    dataset_paths = {
        'co2_data': './data/dataset/ECCO-Darwin_CO2_flux_202001.tif_undefined',
        'methane_data': './data/dataset/methane_emis_microbial_201405.tif_undefined',
        'micasa_data': './data/dataset/MiCASA_v1_ATMC_x3600_y1800_daily_20010101.tif_undefined',
        'oco2_data': './data/dataset/oco2_GEOS_XCO2_L3CO2_day_B10206Ar_20150101.tif_undefined',
        'odic_data': './data/dataset/odiac2023_1km_excl_intl_202208.tif_undefined',
        'gosat_data': './data/dataset/TopDownEmissions_GOSAT_post_coal_GEOS_CHEM_2019.tif_undefined',
        'crop_co2_data': './data/dataset/pilot_topdown_Crop_CO2_Budget_grid_v1_2015.tif_undefined',
        'flux_rh_data': './data/dataset/MiCASAv1_flux_Rh_x3600_y1800_daily_20230615.tif_undefined'
    }

    # Downsample factors for larger datasets
    downscale_factors = {
        'co2_data': 4,
        'methane_data': 4,
        'micasa_data': 4,
        'oco2_data': 4,
        'odic_data': 8,
        'gosat_data': 4,
        'crop_co2_data': 4,
        'flux_rh_data': 4
    }

    # Iterate through datasets and preprocess
    for dataset_name, dataset_path in dataset_paths.items():
        downscale_factor = downscale_factors[dataset_name]
        data = load_tif_data(dataset_path, downscale_factor=downscale_factor)
        data = normalize_data(data)
        save_data(output_folder, f'{dataset_name}.npy', data)

if __name__ == "__main__":
    preprocess_data()
