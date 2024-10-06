import os      
import numpy as np
import rasterio
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt
import time  # For loading spinner
from tensorflow.keras.models import load_model
from geopy.geocoders import Nominatim  # For geocoding
from solve import get_solution_for_issue  # Import solution handling

# Load AI models
@st.cache_resource
def load_models():
    return {
        'co2': load_model('./models/co2_data_prediction_model.keras'),
        'methane': load_model('./models/methane_data_prediction_model.keras'),
        'micasa': load_model('./models/micasa_data_prediction_model.keras'),
        'oco2': load_model('./models/oco2_data_prediction_model.keras'),
        'gosat': load_model('./models/gosat_data_prediction_model.keras'),
        'crop_co2': load_model('./models/crop_co2_data_prediction_model.keras'),
        'flux_rh': load_model('./models/flux_rh_data_prediction_model.keras'),
    }

models = load_models()

# Map emission categories to models
category_to_model = {
    'Greenhouse Gases': 'co2',
    'Human-Based Emissions': 'methane',
    'Nature-Based Emissions': 'micasa',
    'Fossil Fuel Emissions': 'co2',
    'Global CO2 Emission': 'co2',
    'Ocean Absorption': 'crop_co2',
    'Global Methane Emission': 'methane',
    'Atmosphere Methane Concentration': 'methane',
    'Atmosphere CO2 Concentration': 'oco2',
    'Atmosphere NO2 Concentration': 'gosat',
    'Ozone Layer Concentration': 'flux_rh',
    'Heat Wave Prediction': 'micasa'
}

# Auto-select the model based on emission category
def auto_select_model(category_selection):
    if not category_selection:
        return None
    for category in category_selection:
        if category in category_to_model:
            return models[category_to_model[category]]
    return None

# Function to combine multiple .tif files into one dataset for visualization
def combine_tif_files(tif_folder, target_shape=(180, 360)):  # Downsample to a smaller target shape
    combined_data = None
    for root, dirs, files in os.walk(tif_folder):
        for file in files:
            if file.endswith('.tif'):
                filepath = os.path.join(root, file)
                with rasterio.open(filepath) as src:
                    data = src.read(1, out_shape=target_shape, resampling=rasterio.enums.Resampling.bilinear)
                    if combined_data is None:
                        combined_data = data
                    else:
                        combined_data = np.maximum(combined_data, data)  # Combine data using maximum
    return combined_data

# Function to predict emissions using the combined data from .tif files
def predict_emissions(combined_data, model):
    input_shape = model.input_shape
    sequence_length, features = input_shape[1], input_shape[2]

    # Resize combined data to match the model's expected input shape
    resized_data = np.resize(combined_data, (sequence_length, features))

    # Reshape the data to match model input (batch_size, sequence_length, features)
    input_data = resized_data.reshape(1, sequence_length, features)

    # Perform prediction
    predictions = model.predict(input_data)

    # If predictions don't match the combined_data shape, resize them
    predicted_map = np.resize(predictions, combined_data.shape)

    return predicted_map

# Function to create 3D globe map with different styles based on emission category
def plot_emission_globe(predicted_map, lat, lon, location_label, map_style, category, grid_on):
    lat_range = np.linspace(-90, 90, predicted_map.shape[0])
    lon_range = np.linspace(-180, 180, predicted_map.shape[1])

    # Prepare data for Pydeck visualization
    data = []
    for i, lat_v in enumerate(lat_range):
        for j, lon_v in enumerate(lon_range):
            emission_value = float(predicted_map[i, j])
            data.append({
                'lat': float(lat_v),
                'lon': float(lon_v),
                'emission': emission_value
            })

    # Define color scale for heatwaves and different emissions
    if category == "Heat Wave Prediction":
        layer = pdk.Layer(
            'HeatmapLayer',  # Heatmap for heatwave prediction
            data=data,
            get_position='[lon, lat]',
            get_weight='emission',
            radius_pixels=40,
            opacity=0.9
        )
    else:
        layer = pdk.Layer(
            'GridLayer' if grid_on else 'ScatterplotLayer',  # GridLayer for grid on and Scatterplot for grid off
            data=data,
            get_position='[lon, lat]',
            cell_size=50000 if grid_on else None,
            elevation_scale=5 if grid_on else None,
            extruded=grid_on,
            get_fill_color="[255 * emission, 255 * (1 - emission), 0, 150]"
        )

    # Define Pydeck view based on user location
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=5,
        pitch=0,
        bearing=0,
    )

    # Return the Pydeck deck with both layers
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=map_style,
        tooltip={"text": "Emission: {emission}"}
    )

# Function to predict sea level rise and plot the chart
def plot_sea_level_rise():
    years = np.arange(2024, 2034)  # Simulating sea-level rise from 2024-2034
    sea_level_rise = np.cumsum(np.random.normal(0.05, 0.01, len(years)))  # Simulated sea-level rise
    
    # Plot the sea level rise
    fig, ax = plt.subplots()
    ax.plot(years, sea_level_rise, label="Sea Level Rise (m)")
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level Rise (m)')
    ax.set_title('Sea Level Rise Prediction (2024-2034)')
    ax.grid(True)
    st.pyplot(fig)

# Function to generate climate story based on selected data
def generate_climate_story(predicted_emission_map, category_selection):
    story = "## Climate Change Story\n\n"
    story += "In recent years, the world has experienced a rise in greenhouse gas emissions. "
    
    if 'Greenhouse Gases' in category_selection:
        avg_co2_emission = np.mean(predicted_emission_map)
        story += f"The global CO2 emission average is approximately {avg_co2_emission:.2f} tons per capita. "

    if 'Heat Wave Prediction' in category_selection:
        story += "Heat waves are becoming more intense and frequent, impacting both human health and ecosystems. "
        
    if 'Sea Level Rise' in category_selection:
        story += "Rising sea levels due to global warming pose a significant threat to coastal regions. "
        
    story += "\n\nWe must act quickly to address these emissions, implement solutions, and protect vulnerable regions from further damage."
    
    return story

# Function to create an alert box for high emission areas with dynamic thresholds
def alert_over_emission_areas(predicted_map, threshold=0.2):
    lat_range = np.linspace(-90, 90, predicted_map.shape[0])
    lon_range = np.linspace(-180, 180, predicted_map.shape[1])

    over_emission_data = []
    for i, lat_v in enumerate(lat_range):
        for j, lon_v in enumerate(lon_range):
            emission_value = predicted_map[i, j]
            if emission_value > threshold:
                over_emission_data.append({
                    'lat': lat_v,
                    'lon': lon_v,
                    'emission': emission_value
                })

    if over_emission_data:
        st.warning("⚠️ High Emission Areas Detected!")
        st.markdown("<h4 style='color:red;'>Over-Emission Warnings:</h4>", unsafe_allow_html=True)
        for area in over_emission_data:
            st.write(f"📍 Location: Lat: {area['lat']}, Lon: {area['lon']}")
            st.write(f"Emission Value: {area['emission']}")
            issue_key = "high_co2_emission"  # Customize this depending on the emission type
            if st.button(f"Show Solution for Lat: {area['lat']}, Lon: {area['lon']}"):
                description, solutions = get_solution_for_issue(issue_key)
                if description and solutions:
                    st.write(f"**Problem:** {description}")
                    st.write("**Solutions:**")
                    for solution in solutions:
                        st.write(f"- {solution}")
                        
# Sidebar for map style selection
map_style_options = {
    "Satellite": "mapbox://styles/mapbox/satellite-v9",
    "Streets": "mapbox://styles/mapbox/streets-v11",
    "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
    "Dark": "mapbox://styles/mapbox/dark-v10"
}

st.sidebar.subheader("Map Style")
map_style = st.sidebar.selectbox("Select Map Style", list(map_style_options.keys()))

# Sidebar options for emission categories
st.sidebar.header("Emission Categories")
categories = [
    'Greenhouse Gases',
    'Human-Based Emissions',
    'Nature-Based Emissions',
    'Fossil Fuel Emissions',
    'Global CO2 Emission',
    'Ocean Absorption',
    'Global Methane Emission',
    'Atmosphere Methane Concentration',
    'Atmosphere CO2 Concentration',
    'Atmosphere NO2 Concentration',
    'Ozone Layer Concentration',
    'Heat Wave Prediction'
]
category_selection = st.sidebar.multiselect("Select Emission Categories", categories)

# Auto-select the model based on selected categories
model = auto_select_model(category_selection)

# Sidebar for location input
st.sidebar.subheader("Location Options")
location_input = st.sidebar.text_input("Enter location coordinates (lat,lon):", "")

if location_input:
    try:
        lat_lon = location_input.split(",")
        user_lat, user_lon = float(lat_lon[0]), float(lat_lon[1])
        location_label = "User Input Coordinates"
    except ValueError:
        st.error("Please enter valid coordinates in the format 'lat,lon'.")
        user_lat, user_lon = 51.5074, -0.1278  # Default to London
        location_label = "Default (London)"
else:
    user_lat, user_lon = 51.5074, -0.1278  # Default location: London
    location_label = "Default (London)"

# Slider to adjust emission threshold for alerts
threshold = st.sidebar.slider("Emission Alert Threshold", 0.0, 1.0, 0.2)

# Button to toggle gridlines on or off
grid_on = st.sidebar.checkbox("Turn Grid On", value=False)

# Button to show the graph of sea-level rise
if st.sidebar.button("Show Sea-Level Rise Prediction (2024-2034)"):
    plot_sea_level_rise()

# Button to generate the climate story
if st.sidebar.button("Generate Climate Story"):
    if model:
        story = generate_climate_story(combine_tif_files("./data/dataset/"), category_selection)
        st.subheader("Climate Story")
        st.write(story)

# Loading spinner while processing the model prediction
with st.spinner('Processing model prediction...'):
    time.sleep(2)  # Simulate loading time
    # Load data from .tif files
    tif_folder = "./data/dataset/"
    combined_data = combine_tif_files(tif_folder)

# Check if model is selected
if model:
    st.write(f"Predicting emissions using selected model...")
    predicted_emission_map = predict_emissions(combined_data, model)

    # Show different map styles based on the selected emission category
    for category in category_selection:
        st.write(f"Displaying {category} emissions on the map:")
        deck = plot_emission_globe(predicted_emission_map, user_lat, user_lon, location_label, map_style_options[map_style], category, grid_on)
        st.pydeck_chart(deck)

    # Alert box for over-emission areas
    alert_over_emission_areas(predicted_emission_map, threshold)

    # Display selected emission categories
    st.subheader("Selected Emission Categories")
    for category in category_selection:
        st.write(f"- {category}")
else:
    st.error("Please select at least one emission category.")
