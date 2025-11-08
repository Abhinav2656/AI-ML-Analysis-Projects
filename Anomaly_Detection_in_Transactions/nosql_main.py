import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from pymongo import MongoClient
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

client = MongoClient('mongodb://localhost:27017/')

db = client['project_db']  # Database name
collection = db['transactions']  # Collection name (like a table)


JSON_FILE_NAME = 'csvjson.json'
try:
    with open(JSON_FILE_NAME, 'r') as f:
        data_for_mongo = json.load(f)

    # Clear the collection before inserting to avoid duplicates
    collection.delete_many({})

    # Insert the data
    collection.insert_many(data_for_mongo)

    print(f"Successfully inserted {collection.count_documents({})} records into MongoDB.")

except FileNotFoundError:
    print(f"Error: '{JSON_FILE_NAME}' not found.")  # <-- Updated error message
    print("Please make sure the file is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred during MongoDB operations: {e}")
    exit()


print("Loading data from MongoDB into pandas DataFrame...")
cursor = collection.find()
data = pd.DataFrame(list(cursor))


if '_id' in data.columns:
    data = data.drop('_id', axis=1)

print("Data loaded successfully. Showing head:")
print(data.head())


# Distribution of transaction Amount
fig_amount = px.histogram(data, x='Transaction_Amount',
                          nbins=20,
                          title='Distribution of Transaction Amounts')
fig_amount.show()  # <-- CHANGED

# Transaction amount by Account Type
fig_box_amount = px.box(data, x='Account_Type',
                        y='Transaction_Amount', title='Transaction Amount by Account Type')
fig_box_amount.show()  # <-- CHANGED

# Average Transaction Amount VS Age
fig_scatter_avg_amount_age = px.scatter(data, x='Age',
                                        y='Average_Transaction_Amount',
                                        color='Account_Type',
                                        title='Average Transaction Amount vs. Age',
                                        trendline='ols')
fig_scatter_avg_amount_age.show()  # <-- CHANGED

# Count of Transactions by Day of the week
fig_day_of_week = px.bar(data, x='Day_of_Week',
                         title='Count of Transactions by Day of the Week')
fig_day_of_week.show()  # <-- CHANGED



# Isolating Anomalies
anomaly_threshold = data['Transaction_Amount'].quantile(0.99)
data['Is_Anomaly'] = (data['Transaction_Amount'] > anomaly_threshold).astype(int)

fig_anomalies = px.scatter(data, x='Transaction_Amount',
                           y='Average_Transaction_Amount',
                           color='Is_Anomaly', title='Anomalies in Transaction Amount')
fig_anomalies.update_traces(marker=dict(size=12), selector=dict(mode='markers', marker_size=1))
fig_anomalies.show()  # <-- CHANGED

# Calculate ratio of anomalies
num_anomalies = data['Is_Anomaly'].sum()
total_instances = data.shape[0]
anomaly_ratio = num_anomalies / total_instances
print(f"Anomaly Ratio: {anomaly_ratio}")

# ML Model for detecting anomalies
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

X = data[relevant_features]
y = data['Is_Anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)

# Model Report/Performance
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]

report = classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly'])
print("Classification Report:")
print(report)
print("\n--- Analysis Complete ---")


print("\n--- Live Anomaly Prediction ---")
# User Inputs for Model
user_inputs = []
for feature in relevant_features:
    user_input = float(input(f"Enter the value for '{feature}': "))
    user_inputs.append(user_input)

user_df = pd.DataFrame([user_inputs], columns=relevant_features)
user_anomaly_pred = model.predict(user_df)


user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0

if user_anomaly_pred_binary == 1:
    print("\nAnomaly detected: This transaction is flagged as an anomaly")
else:
    print("\nNo anomaly detected: This transaction is normal")


print("\n--- Analysis Complete ---")