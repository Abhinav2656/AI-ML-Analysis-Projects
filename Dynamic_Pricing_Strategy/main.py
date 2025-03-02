import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dynamic_pricing.csv")


# Exploratory Data Analysis

fig = px.scatter(data, x='Expected_Ride_Duration',
                 y='Historical_Cost_of_Ride',
                 title='Expected Ride Duration vs. Historical Cost of Ride',
                 trendline='ols')
fig.show()

fig = px.box(data, x="Vehicle_Type", y='Historical_Cost_of_Ride',
             title='Historical Cost of Ride Distribution by Vehicle Type')
fig.show()

# corr_matrix = data.corr()  (Error in getting correlation matrix)
# fig = go.Figure(data.go.Heatmap(z=corr_matrix.values,
#                                 x=corr_matrix.columns,
#                                 y=corr_matrix.columns,
#                                 colorscale='Viridis'))
# fig.update_layout(title='Correlation Matrix')
# fig.show()


# Implementing a Dynamic Pricing Strategy
high_demand_percentile = 75
low_demand_percentile = 25
data['demand_multiplier'] = np.where(data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                     data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                     data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile))

high_supply_percentile = 75
low_supply_percentile = 25
data['supply_multiplier'] = np.where(data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], low_supply_percentile),
                                     np.percentile(data['Number_of_Drivers'], high_supply_percentile) / data['Number_of_Drivers'],
                                     np.percentile(data['Number_of_Drivers'], low_supply_percentile) / data['Number_of_Drivers'])

demand_threshold_high = 1.2
demand_threshold_low = 0.8
supply_threshold_high = 0.8
supply_threshold_low = 1.2

data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
    np.maximum(data['demand_multiplier'], demand_threshold_low) *
    np.maximum(data['supply_multiplier'], demand_threshold_high)
)


# Profit percentage after implementing dynamic pricing strategy
data['profit_percentage'] = ((data['adjusted_ride_cost'] - data['Historical_Cost_of_Ride']) / data['Historical_Cost_of_Ride']) * 100
profitable_rides = data[data['profit_percentage'] > 0]

loss_rides = data[data['profit_percentage'] < 0]

profitable_count = len(profitable_rides)
loss_count = len(loss_rides)

labels = ['Profitable_Rides', 'Loss Rides']
values = [profitable_count, loss_count]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
fig.update_layout(title='Profitability of Rides (Dynamic Priciing vs. Historical Pricing)')
fig.show()

fig = px.scatter(data, x='Expected_Ride_Duration', y='adjusted_ride_cost',
                 title='Expected Ride Duration vs. Cost of Ride', trendline='ols')
fig.show()


def data_preprocessing_pipeline(data):

    numeric_features = data.select_dtypes(include=['float', 'int']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

    for feature in numeric_features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        data[feature] = np.where((data[feature] < lower_bound) | (data[feature] > upper_bound),
                                 data[feature].mean(), data[feature])

    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

    return data

data['Vehicle_Type'] = data['Vehicle_Type'].map({'Premium': 1, "Economy": 0})


# splitting data
x = np.array(data[['Number_of_Riders', 'Number_of_Drivers', 'Vehicle_Type', 'Expected_Ride_Duration']])
y = np.array(data[['adjusted_ride_cost']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()


# Training random forest regression model
model = RandomForestRegressor()
model.fit(x_train, y_train)


def get_vehicle_type_numeric(vehicle_type):
    vehicle_type_mapping = {
        "Premium": 1, "Economy": 0
    }
    vehicle_type_numeric = vehicle_type_mapping.get(vehicle_type)
    return vehicle_type_numeric

def predict_price(number_of_riders, number_of_drivers, vehicle_type, Expected_Ride_Duration):
    vehicle_type_numeric = get_vehicle_type_numeric(vehicle_type)
    if vehicle_type_numeric is None:
        raise ValueError("Invalid vehicle type")
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, Expected_Ride_Duration]])
    predicted_price = model.predict(input_data)
    return predicted_price

user_number_of_riders = 50
user_number_of_drivers = 25
user_vehicle_type = "Economy"
Expected_Ride_Duration = 30
predicted_price = predict_price(user_number_of_riders, user_number_of_drivers, user_vehicle_type, Expected_Ride_Duration)
print("Predicted price: ", predicted_price)


y_pred = model.predict(x_test)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x = y_test.flatten(), y=y_pred,
    mode='markers', name='Actual vs Predicted'
))

fig.add_trace(go.Scatter(
    x=[min(y_test.flatten()), max(y_test.flatten())],
    y=[min(y_test.flatten()), max(y_test.flatten())],
    mode='lines', name='Ideal', line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Actual Values', yaxis_title='Predicted Values',
    showlegend=True
)
fig.show()
