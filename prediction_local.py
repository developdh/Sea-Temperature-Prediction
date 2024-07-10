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

# Step 7: combine the lat*lon points into a single dataset and visualize the results
combined_results = xr.Dataset(
    {
        "predicted_temperature": (("lat", "lon"), np.concatenate(y_pred_list)),
        "actual_temperature": (("lat", "lon"), np.concatenate(y_test_list)),
        # "mse": (("lat", "lon"), mse_list),
    },
    coords={
        "lat": target.lat.values,
        "lon": target.lon.values,
    },
)

# Convert numpy arrays to DataArrays
combined_results["predicted_temperature"] = xr.DataArray(combined_results["predicted_temperature"], dims=("lat", "lon"))
combined_results["actual_temperature"] = xr.DataArray(combined_results["actual_temperature"], dims=("lat", "lon"))

# Visualize the results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
combined_results["predicted_temperature"].plot(ax=axes[0])
combined_results["actual_temperature"].plot(ax=axes[1])
axes[0].set_title("Predicted Temperature")
axes[1].set_title("Actual Temperature")
plt.tight_layout()
plt.show()