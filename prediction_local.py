import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
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

# Initialize lists to store the combined results
predicted_temp_combined = np.zeros((target.shape[0], target.shape[1], target.shape[2]))
actual_temp_combined = np.zeros((target.shape[0], target.shape[1], target.shape[2]))

# Split the data into training and testing sets for each lat*lon point and train models
rf_models = []
mse_list = []

for lat in range(target.shape[1]):
    for lon in range(target.shape[2]):
        X_train, X_test, y_train, y_test = train_test_split(features.values, target.values[:, lat, lon], test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
        
        # Predict on the entire dataset for visualization purposes
        y_full_pred = model.predict(features.values)
        predicted_temp_combined[:, lat, lon] = y_full_pred
        actual_temp_combined[:, lat, lon] = target.values[:, lat, lon]

# Combine the results into a single dataset
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

# Print Mean Squared Error for reference
print("Mean Squared Error List:", mse_list)

# Visualize the results for each time period
for t in range(combined_results["time"].shape[0]):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    combined_results["predicted_temperature"].isel(time=t).plot(ax=axes[0])
    combined_results["actual_temperature"].isel(time=t).plot(ax=axes[1])
    axes[0].set_title(f"Predicted Temperature at time={t}")
    axes[1].set_title(f"Actual Temperature at time={t}")
    plt.tight_layout()
    plt.show()
