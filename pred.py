# Step 1: Import necessary libraries
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import nc_time_axis
import xgboost as xgb

# Step 2: Load the netCDF datasets

ocean_current_data_vo_surface = xr.open_dataset('_data/_ocean_current_data_vo_surface.nc')
ocean_current_data_uo_surface = xr.open_dataset('_data/_ocean_current_data_uo_surface.nc')
temperature_data = xr.open_dataset('_data/_temperature_data.nc')
psl_data = xr.open_dataset('_data/_psl_data.nc')

# print(ocean_current_data_vo_surface['time'].shape)

# ocean_current_data_vo_surface['vo'].isel(time=0).plot()
# plt.show()
# ocean_current_data_uo_surface['uo'].isel(time=0).plot()
# plt.show()
# temperature_data['tos'].isel(time=0).plot()
# plt.show()
# psl_data['psl'].isel(time=0).plot()
# plt.show()

avg_ocean_current_data_vo_surface = ocean_current_data_vo_surface.mean(dim=('lat', 'lon'))
avg_ocean_current_data_uo_surface = ocean_current_data_uo_surface.mean(dim=('lat', 'lon'))
avg_psl_data = psl_data.mean(dim=('lat', 'lon'))
avg_temperature_data = temperature_data.mean(dim=('lat', 'lon'))

# avg_ocean_current_data_vo_surface['vo'].plot()
# plt.show()
# avg_ocean_current_data_uo_surface['uo'].plot()
# plt.show()
# avg_psl_data['psl'].plot()
# plt.show()
# avg_temperature_data['tos'].plot()
# plt.show()

mean_avg_ocean_current_data_vo_surface = avg_ocean_current_data_vo_surface.mean(dim='time')
sd_avg_ocean_current_data_vo_surface = avg_ocean_current_data_vo_surface.std(dim='time')
# print(mean_avg_ocean_current_data_vo_surface)
# print(sd_avg_ocean_current_data_vo_surface)

mean_avg_ocean_current_data_uo_surface = avg_ocean_current_data_uo_surface.mean(dim='time')
sd_avg_ocean_current_data_uo_surface = avg_ocean_current_data_uo_surface.std(dim='time')
# print(mean_avg_ocean_current_data_uo_surface)
# print(sd_avg_ocean_current_data_uo_surface)

mean_avg_psl_data = avg_psl_data.mean(dim='time')
sd_avg_psl_data = avg_psl_data.std(dim='time')
# print(mean_avg_psl_data)
# print(sd_avg_psl_data)

mean_avg_temperature_data = avg_temperature_data.mean(dim='time')
sd_avg_temperature_data = avg_temperature_data.std(dim='time')
# print(mean_avg_temperature_data)
# print(sd_avg_temperature_data)

avg_ocean_current_data_vo_surface_normalized = (avg_ocean_current_data_vo_surface - mean_avg_ocean_current_data_vo_surface) / sd_avg_ocean_current_data_vo_surface
avg_ocean_current_data_uo_surface_normalized = (avg_ocean_current_data_uo_surface - mean_avg_ocean_current_data_uo_surface) / sd_avg_ocean_current_data_uo_surface
avg_psl_data_normalized = (avg_psl_data - mean_avg_psl_data) / sd_avg_psl_data
avg_temperature_data_normalized = (avg_temperature_data - mean_avg_temperature_data) / sd_avg_temperature_data

# avg_ocean_current_data_vo_surface['vo'].plot()
# plt.show()
# avg_ocean_current_data_vo_surface_normalized['vo'].plot()
# # plt.legend()
# plt.show()

# print(avg_ocean_current_data_uo_surface_normalized.dims)
# print(avg_ocean_current_data_vo_surface_normalized.dims)
# print(avg_psl_data_normalized.dims)
# print(avg_temperature_data_normalized.dims)


# Step 3: Prepare the data for training
X = pd.concat([avg_ocean_current_data_vo_surface_normalized['vo'].to_dataframe(), avg_ocean_current_data_uo_surface_normalized['uo'].to_dataframe(), avg_psl_data_normalized['psl'].to_dataframe()], axis=1)
X = X.apply(pd.to_numeric).fillna(0)

print(X)

y = avg_temperature_data_normalized['tos']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#RANDOM FOREST REGRESSOR

# Step 5: Train the random forest regressor model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 8: Visualize the predicted values and the actual values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted (Random Forest)')
plt.xlabel('Sample')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.show()



#MLP REGRESSOR

# Step 9: Train the MLP regressor model
mlp_model = MLPRegressor()
mlp_model.fit(X_train, y_train)

# Step 10: Make predictions on the test set
y_pred_mlp = mlp_model.predict(X_test)

# Step 11: Evaluate the model
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print("Mean Squared Error (MLP):", mse_mlp)

# Step 12: Visualize the predicted values and the actual values for MLP model
plt.plot(y_test, label='Actual')
plt.plot(y_pred_mlp, label='Predicted (MLP)')
plt.xlabel('Sample')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.show()


# XGBoost REGRESSOR

# Step 13: Train the XGBoost regressor model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train.values, y_train.values)

# Step 14: Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test.values)

# Step 15: Evaluate the model
mse_xgb = mean_squared_error(y_test.values, y_pred_xgb)
print("Mean Squared Error (XGBoost):", mse_xgb)

# Step 16: Visualize the predicted values and the actual values for XGBoost model
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_xgb, label='Predicted (XGBoost)')
plt.xlabel('Sample')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.show()
