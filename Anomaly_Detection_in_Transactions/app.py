import streamlit as st
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

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(page_title="Anomaly Detector", layout="wide")
st.title("💳 Financial Transaction Anomaly Detector")


# ====================================================================
# "ENGINE" - Cached to run only ONCE
# ====================================================================
# This "decorator" is the magic of Streamlit.
# It tells Streamlit to run this function only once and save the results.
@st.cache_data
def load_data_and_train_model():
    """
    Connects to MongoDB, loads data, and trains the model.
    This function is cached, so it only runs once.
    """
    # --- 1. CONNECT TO MONGODB AND LOAD DATA ---
    client = MongoClient('mongodb://localhost:27017/')
    db = client['project_db']
    collection = db['transactions']

    # Load and insert data (this part will only run once on first load)
    JSON_FILE_NAME = 'csvjson.json'
    try:
        with open(JSON_FILE_NAME, 'r') as f:
            data_for_mongo = json.load(f)
        collection.delete_many({})
        collection.insert_many(data_for_mongo)
    except Exception as e:
        # If Mongo is down, try to load from the CSV as a fallback
        st.error(f"MongoDB error: {e}. Falling back to CSV.")
        try:
            data = pd.read_csv("transaction_anomalies_dataset.csv")
        except FileNotFoundError:
            st.error("FATAL: Neither MongoDB nor transaction_anomalies_dataset.csv could be loaded.")
            return None, None, None, None, None

    # Read data FROM MongoDB into a pandas DataFrame
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))
    if '_id' in data.columns:
        data = data.drop('_id', axis=1)

    # --- 2. CREATE PLOTS (but don't show them) ---
    # We create the plot "objects" here. The UI will decide when to show them.
    fig_amount = px.histogram(data, x='Transaction_Amount', nbins=20, title='Distribution of Transaction Amounts')
    fig_box_amount = px.box(data, x='Account_Type', y='Transaction_Amount', title='Transaction Amount by Account Type')
    fig_scatter_avg_amount_age = px.scatter(data, x='Age', y='Average_Transaction_Amount', color='Account_Type',
                                            title='Average Transaction Amount vs. Age', trendline='ols')
    fig_day_of_week = px.bar(data, x='Day_of_Week', title='Count of Transactions by Day of the Week')

    # Isolating Anomalies
    anomaly_threshold = data['Transaction_Amount'].quantile(0.99)
    data['Is_Anomaly'] = (data['Transaction_Amount'] > anomaly_threshold).astype(int)
    fig_anomalies = px.scatter(data, x='Transaction_Amount', y='Average_Transaction_Amount', color='Is_Anomaly',
                               title='Anomalies in Transaction Amount')

    # Calculate ratio of anomalies
    num_anomalies = data['Is_Anomaly'].sum()
    total_instances = data.shape[0]
    anomaly_ratio = num_anomalies / total_instances

    # --- 3. TRAIN ML MODEL ---
    relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
    X = data[relevant_features]
    y = data['Is_Anomaly']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use the tuned contamination value
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train)

    # Model Report/Performance
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]
    report = classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly'])

    # Pack all plots into a dictionary
    plots = {
        "dist": fig_amount, "box": fig_box_amount, "scatter_age": fig_scatter_avg_amount_age,
        "bar_day": fig_day_of_week, "anomalies": fig_anomalies
    }

    return model, relevant_features, report, anomaly_ratio, plots


# --- Load all data and models (this uses the cache) ---
model, relevant_features, report, anomaly_ratio, plots = load_data_and_train_model()

# ====================================================================
# USER INTERFACE (The App)
# ====================================================================

# --- Part 1: Live Anomaly Prediction (The main app) ---
st.header("🕵️ Live Anomaly Prediction")
st.write("Enter the details of a transaction to check if it's an anomaly.")

# Use columns for a cleaner layout
col1, col2, col3 = st.columns(3)
with col1:
    user_tx_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0, step=50.0)
with col2:
    user_avg_amount = st.number_input("User's Avg. Transaction Amount ($)", min_value=0.0, value=950.0, step=50.0)
with col3:
    user_freq = st.number_input("User's Transaction Frequency (per month)", min_value=1, value=10, step=1)

# Prediction button
if st.button("Check Transaction", type="primary"):
    # Create the DataFrame for the model
    user_inputs = [user_tx_amount, user_avg_amount, user_freq]
    user_df = pd.DataFrame([user_inputs], columns=relevant_features)

    # Run the prediction
    user_anomaly_pred = model.predict(user_df)
    user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0

    # Display the result
    if user_anomaly_pred_binary == 1:
        st.error("🔴 Anomaly Detected: This transaction is flagged as suspicious!")
    else:
        st.success("✅ No Anomaly Detected: This transaction appears normal.")

# --- Part 2: Model Performance ---
st.header("🤖 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Anomaly Ratio (in Training Data)", f"{anomaly_ratio * 100:.2f}%")
with col2:
    st.write("Classification Report (on Test Data)")
    # Use st.text to preserve the formatting of the report
    st.text(report)

# --- Part 3: Data Analysis Plots ---
with st.expander("📊 Show Data Analysis Plots"):
    st.write("These plots show the analysis of the full dataset used to train the model.")

    # Display all the plots
    st.plotly_chart(plots['anomalies'], use_container_width=True)
    st.plotly_chart(plots['dist'], use_container_width=True)
    st.plotly_chart(plots['box'], use_container_width=True)
    st.plotly_chart(plots['scatter_age'], use_container_width=True)
    st.plotly_chart(plots['bar_day'], use_container_width=True)