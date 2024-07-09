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
import tensorflow as tf

# Step 2: Load the netCDF datasets

ocean_current_data_vo_surface = xr.open_dataset('_data/_ocean_current_data_vo_surface.nc')
ocean_current_data_uo_surface = xr.open_dataset('_data/_ocean_current_data_uo_surface.nc')
temperature_data = xr.open_dataset('_data/_temperature_data.nc')
psl_data = xr.open_dataset('_data/_psl_data.nc')

# Step 3: Prepare the data for training

avg_ocean_current_data_vo_surface = ocean_current_data_vo_surface.mean(dim=('lat', 'lon'))
avg_ocean_current_data_uo_surface = ocean_current_data_uo_surface.mean(dim=('lat', 'lon'))
avg_psl_data = psl_data.mean(dim=('lat', 'lon'))
avg_temperature_data = temperature_data.mean(dim=('lat', 'lon'))

mean_avg_ocean_current_data_vo_surface = avg_ocean_current_data_vo_surface.mean(dim='time')
sd_avg_ocean_current_data_vo_surface = avg_ocean_current_data_vo_surface.std(dim='time')

mean_avg_ocean_current_data_uo_surface = avg_ocean_current_data_uo_surface.mean(dim='time')
sd_avg_ocean_current_data_uo_surface = avg_ocean_current_data_uo_surface.std(dim='time')

mean_avg_psl_data = avg_psl_data.mean(dim='time')
sd_avg_psl_data = avg_psl_data.std(dim='time')

mean_avg_temperature_data = avg_temperature_data.mean(dim='time')
sd_avg_temperature_data = avg_temperature_data.std(dim='time')

avg_ocean_current_data_vo_surface_normalized = (avg_ocean_current_data_vo_surface - mean_avg_ocean_current_data_vo_surface) / sd_avg_ocean_current_data_vo_surface
avg_ocean_current_data_uo_surface_normalized = (avg_ocean_current_data_uo_surface - mean_avg_ocean_current_data_uo_surface) / sd_avg_ocean_current_data_uo_surface
avg_psl_data_normalized = (avg_psl_data - mean_avg_psl_data) / sd_avg_psl_data
avg_temperature_data_normalized = (avg_temperature_data - mean_avg_temperature_data) / sd_avg_temperature_data

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


#TENSORFLOW REGRESSOR

# Step 17: Convert the data to TensorFlow tensors
X_train_tf = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Step 18: Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Step 19: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 20: Train the model
model.fit(X_train_tf, y_train_tf, epochs=10, batch_size=32)

# Step 21: Evaluate the model
mse_tf = model.evaluate(X_test_tf, y_test_tf)
print("Mean Squared Error (TensorFlow):", mse_tf)

# Step 22: Make predictions on the test set
y_pred_tf = model.predict(X_test_tf)

# Step 23: Visualize the predicted values and the actual values for TensorFlow model
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_tf, label='Predicted (TensorFlow)')
plt.xlabel('Sample')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.show()


#RESULTS

# Step 24: Compare the performance of the models
models = ['Random Forest', 'MLP', 'XGBoost', 'TensorFlow']
mse_scores = [mse, mse_mlp, mse_xgb, mse_tf]

best_model_index = mse_scores.index(min(mse_scores))
best_model = models[best_model_index]
best_mse = mse_scores[best_model_index]

print("Best Model:", best_model)
print("Best Mean Squared Error:", best_mse)