import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
import pickle
import nc_time_axis
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the netCDF datasets
ocean_current_data_vo_surface = xr.open_dataset('_data/_ocean_current_data_vo_surface.nc')
ocean_current_data_uo_surface = xr.open_dataset('_data/_ocean_current_data_uo_surface.nc')
temperature_data = xr.open_dataset('_data/_temperature_data.nc')
psl_data = xr.open_dataset('_data/_psl_data.nc')

# Step 3: Prepare the data for training and testing
# Combine the features into a single dataframe
features = pd.concat([ocean_current_data_vo_surface.to_dataframe().reset_index(drop=True), 
                      ocean_current_data_uo_surface.to_dataframe().reset_index(drop=True), 
                      psl_data.to_dataframe().reset_index(drop=True)], axis=1)
features = features[:768]  # Limit the number of samples to match the target variable

# Extract the target variable
target = temperature_data['tos']

# Split the data into training and testing sets for each lat*lon point
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for lat in range(target.shape[1]):
    for lon in range(target.shape[2]):
        X_train, X_test, y_train, y_test = train_test_split(features.values, target.values[:, lat, lon], test_size=0.01, random_state=42)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

# Step 4: Load the model for each lat*lon point
rf_models = []

with open("local_model/randomforest.pckl", "rb") as f:
    while True:
        try:
            rf_models.append(pickle.load(f))
        except EOFError:
            break

# Step 5: Make predictions for every lat*lon point
y_pred_list = []

for i in range(len(X_train_list)):
    y_pred = rf_models[i].predict(X_test_list[i])
    y_pred_list.append(y_pred)

# Step 6: Evaluate the model for each lat*lon point
mse_list = []

for i in range(len(y_train_list)):
    mse = mean_squared_error(y_test_list[i], y_pred_list[i])
    mse_list.append(mse)

# Step 7: Combine the lat*lon points into a single dataset and visualize the results
lat_dim = target.shape[1]
lon_dim = target.shape[2]
time_dim = target.shape[0]  # Assuming the first dimension is time

# Initialize empty arrays to store combined results
predicted_temp_combined = np.zeros((time_dim, lat_dim, lon_dim))
actual_temp_combined = np.zeros((time_dim, lat_dim, lon_dim))

index = 0
for lat in range(lat_dim):
    for lon in range(lon_dim):
        # Make sure to only assign the predictions for the available time points
        available_time_points = len(y_pred_list[index])
        predicted_temp_combined[:available_time_points, lat, lon] = y_pred_list[index]
        actual_temp_combined[:available_time_points, lat, lon] = y_test_list[index]
        index += 1

combined_results = xr.Dataset(
    {
        "predicted_temperature": (("time", "lat", "lon"), predicted_temp_combined),
        "actual_temperature": (("time", "lat", "lon"), actual_temp_combined),
    },
    coords={
        "time": target.time.values,
        "lat": target.lat.values,
        "lon": target.lon.values,
    },
)

# Visualize the results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
combined_results["predicted_temperature"].isel(time=0).plot(ax=axes[0])
combined_results["actual_temperature"].isel(time=0).plot(ax=axes[1])
axes[0].set_title("Predicted Temperature")
axes[1].set_title("Actual Temperature")
plt.tight_layout()
plt.show()
