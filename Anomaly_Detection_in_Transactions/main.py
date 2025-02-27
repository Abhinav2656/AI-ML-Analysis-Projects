import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import IsolationForest
from sklearn.metrics import classification_report

data = pd.read_csv("transaction_anomalies_dataset.csv")


# Distribution of transaction Amount
fig_amount = px.histogram(data, x='Transaction_Amount',
                          nbins=20,
                          title='Distribution of Transaction Amounts')
fig_amount.show()


# Transaction amount by Account Type
fig_box_amount = px.box(data, x='Account_Type',
                        y='Transaction_Amount', title='Transaction Amount by Account Type')
fig_box_amount.show()


# Average Transaction Amount VS Age
fig_scatter_avg_amount_age = px.scatter(data, x='Age',
                                        y='Average_Transaction_Amount',
                                        color='Account_Type',
                                        title='Average Transaction Amount vs. Age',
                                        trendline='ols')
fig_scatter_avg_amount_age.show()


# Count of Transactions by  Day of the week
fig_day_of_week = px.bar(data, x='Day_of_Week',
                         title='Coutn of Transactions by Day of the Week')
fig_day_of_week.show()


# Correlation Heatmap(Error in converting string to float)
# correlation_matrix = data.corr()
# fig_corr_heatmap = px.imshow(correlation_matrix, title='Correlation Heatmap')
#
# fig_corr_heatmap.show()


# Anomalies in Transaction Amount
mean_amount= data['Transaction_Amount'].mean()
std_amount = data['Transaction_Amount'].std()

anomaly_threshold = mean_amount + 2 * std_amount

data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold

fig_anomalies = px.scatter(data, x='Transaction_Amount', y='Average_Transaction_Amount',
                           color='Is_Anomaly', title='Anomalies in Transaction Amount')
fig_anomalies.update_traces(marker=dict(size=12), selector=dict(mode='markers', marker_size=1))
fig_anomalies.show()

# Calculate ratio of anomalies
num_anomalies = data['Is_Anomaly'].sum()
total_instances = data.shape[0]
anomaly_ratio = num_anomalies / total_instances
print(anomaly_ratio)

# ML Model for detecting anomalies
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

X=data[relevant_features]
y=data['Is_Anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_train)

# Model Report/Performance
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]

report = classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly'])
print(report)

# User Inputs for Model
user_inputs = []
for feature in relevant_features:
    user_input = float(input(f"Enter the value for '{feature}': "))
    user_inputs.append(user_input)

user_df = pd.DataFrame([user_inputs], columns=relevant_features)
user_anomaly_pred = model.predict(user_df)
user_anomaly_pred_binary = i if user_anomaly_pred == -1 else 0

if user_anomaly_pred_binary == 1:
    print("Anomaly detected: This transaction is flagged as an anomaly")
else:
    print("No anomaly detected: This transaction is normal")
