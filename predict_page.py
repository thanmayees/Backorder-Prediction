import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
import sqlite3

# Function to load the models
@st.cache_resource
def load_model():
    try:
        with open('saved_models.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error("The saved models file was not found. Please ensure 'saved_models.pkl' is in the current directory.")
        return None

# Function to initialize the SQLite database
def init_db():
    conn = sqlite3.connect('backorder_predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku INTEGER,
            national_inv INTEGER,
            lead_time INTEGER,
            in_transit_qty INTEGER,
            forecast_3_month INTEGER,
            forecast_6_month INTEGER,
            forecast_9_month INTEGER,
            sales_1_month INTEGER,
            sales_3_month INTEGER,
            sales_6_month INTEGER,
            sales_9_month INTEGER,
            min_bank INTEGER,
            potential_issue INTEGER,
            pieces_past_due INTEGER,
            perf_6_month_avg REAL,
            perf_12_month_avg REAL,
            local_bo_qty INTEGER,
            deck_risk INTEGER,
            oe_constraint INTEGER,
            ppap_risk INTEGER,
            stop_auto_buy INTEGER,
            rev_stop INTEGER,
            logistic_regression_prediction INTEGER,
            decision_tree_prediction INTEGER,
            kmeans_prediction INTEGER,
            random_forest_prediction INTEGER,
            hard_vote_result TEXT
        )
    ''')
    conn.commit()
    return conn

# Function to save prediction results to the SQLite database
def save_prediction_to_db(new_data, predictions, hard_vote_result):
    conn = sqlite3.connect('backorder_predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            sku, national_inv, lead_time, in_transit_qty, forecast_3_month,
            forecast_6_month, forecast_9_month, sales_1_month, sales_3_month,
            sales_6_month, sales_9_month, min_bank, potential_issue, pieces_past_due,
            perf_6_month_avg, perf_12_month_avg, local_bo_qty, deck_risk, oe_constraint,
            ppap_risk, stop_auto_buy, rev_stop, logistic_regression_prediction,
            decision_tree_prediction, kmeans_prediction, random_forest_prediction,
            hard_vote_result
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        new_data['sku'], new_data['national_inv'], new_data['lead_time'], new_data['in_transit_qty'],
        new_data['forecast_3_month'], new_data['forecast_6_month'], new_data['forecast_9_month'],
        new_data['sales_1_month'], new_data['sales_3_month'], new_data['sales_6_month'],
        new_data['sales_9_month'], new_data['min_bank'], new_data['potential_issue'],
        new_data['pieces_past_due'], new_data['perf_6_month_avg'], new_data['perf_12_month_avg'],
        new_data['local_bo_qty'], new_data['deck_risk'], new_data['oe_constraint'],
        new_data['ppap_risk'], new_data['stop_auto_buy'], new_data['rev_stop'],
        predictions.get('logistic_regression'), predictions.get('decision_tree'),
        predictions.get('kmeans'), predictions.get('random_forest'),
        hard_vote_result
    ))
    conn.commit()
    conn.close()

# Function to display prediction results from the SQLite database
def display_prediction_db():
    conn = sqlite3.connect('backorder_predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    st.subheader("Database of Predictions")
    st.dataframe(df)
    conn.close()

# Load models
data = load_model()
if data:
    logistic_regression_model = data["logistic_model"]
    decision_tree_model = data["decision_tree_model"]
    kmeans_model = data["kmeans_model"]
    random_forest_model = data["random_forest_model"]
else:
    logistic_regression_model = decision_tree_model = kmeans_model = random_forest_model = None

# Function to make predictions
def predict_backorder(new_data_row):
    predictions = {}

    if logistic_regression_model:
        logistic_regression_prediction = logistic_regression_model.predict([new_data_row.drop('sku').values])[0]
        predictions['logistic_regression'] = logistic_regression_prediction

    if decision_tree_model:
        decision_tree_prediction = decision_tree_model.predict([new_data_row.drop('sku').values])[0]
        predictions['decision_tree'] = decision_tree_prediction

    if kmeans_model:
        new_data_pca = kmeans_model.named_steps['pca'].transform([new_data_row.drop('sku').values])
        kmeans_prediction = kmeans_model.named_steps['kmeans'].predict(new_data_pca)[0]
        predictions['kmeans'] = kmeans_prediction

    if random_forest_model:
        random_forest_prediction = random_forest_model.predict([new_data_row.drop('sku').values])[0]
        predictions['random_forest'] = random_forest_prediction

    return predictions

# Function to perform hard voting
def hard_voting_classifier(predictions):
    votes = list(predictions.values())
    majority_vote = np.bincount(votes).argmax()
    return "GOING TO BACKORDER" if majority_vote == 1 else "NOT GOING TO BACKORDER"

# Streamlit app
def show_predict_page():
    st.markdown(
        """
        <style>
        .full-app-container {
            background: linear-gradient(to right, #D3959B, #BFE6BA); 
            width: 100%;
            height: 100%;
            position: fixed;
            top: 0;
            left: 0;
            overflow: auto;
        }
        .predict_button {
            width: 200px;
            height: 60px;
            font-size: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="full-app-container">', unsafe_allow_html=True)

    st.title("Predicting the Backorder")
    st.write("This is a web app to predict the backorder of a product based on the values you give")

    st.subheader("Input Data")

    col1, col2 = st.columns(2)

    new_data = {}

    with col1:
        new_data['sku'] = st.number_input("SKU", min_value=0)
        new_data['national_inv'] = st.number_input("National Inventory", min_value=0)
        new_data['lead_time'] = st.number_input("Lead Time", min_value=0)
        new_data['in_transit_qty'] = st.number_input("In-Transit Quantity", min_value=0)
        new_data['forecast_3_month'] = st.number_input("Forecast for 3 Months", min_value=0)
        new_data['forecast_6_month'] = st.number_input("Forecast for 6 Months", min_value=0)
        new_data['forecast_9_month'] = st.number_input("Forecast for 9 Months", min_value=0)
        new_data['sales_1_month'] = st.number_input("Sales for 1 Month", min_value=0)
        new_data['sales_3_month'] = st.number_input("Sales for 3 Months", min_value=0)
        new_data['sales_6_month'] = st.number_input("Sales for 6 Months", min_value=0)
        new_data['local_bo_qty'] = st.number_input("Local Backorder Quantity", min_value=0)
    
    with col2:
        new_data['sales_9_month'] = st.number_input("Sales for 9 Months", min_value=0)
        new_data['min_bank'] = st.number_input("Minimum Bank", min_value=0)
        new_data['potential_issue'] = st.selectbox("Potential Issue", [0, 1])
        new_data['pieces_past_due'] = st.number_input("Pieces Past Due", min_value=0)
        new_data['perf_6_month_avg'] = st.number_input("Performance Average for 6 Months", min_value=0.0)
        new_data['perf_12_month_avg'] = st.number_input("Performance Average for 12 Months", min_value=0.0)
        new_data['deck_risk'] = st.selectbox("Deck Risk", [0, 1])
        new_data['oe_constraint'] = st.selectbox("OE Constraint", [0, 1])
        new_data['ppap_risk'] = st.selectbox("PPAP Risk", [0, 1])
        new_data['stop_auto_buy'] = st.selectbox("Stop Auto Buy", [0, 1])
        new_data['rev_stop'] = st.selectbox("Revenue Stop", [0, 1])

    if st.button("Predict", key="predict_button", help="Click to predict whether the product will go to backorder or not"):
        new_data_row = pd.Series(new_data)
        predictions = predict_backorder(new_data_row)

        st.subheader("Predictions")
        if predictions:
            for model, prediction in predictions.items():
                prediction_sentence = "GOING TO BACKORDER" if prediction == 1 else "NOT GOING TO BACKORDER"
                st.write(f"{model.replace('_', ' ').title()} Prediction:", prediction_sentence)
            
            hard_vote_result = hard_voting_classifier(predictions)
            st.subheader("Hard Voting Classifier Prediction")
            st.write(hard_vote_result)
            
            save_prediction_to_db(new_data, predictions, hard_vote_result)
        else:
            st.write("No models are loaded to make predictions.")

    st.markdown('</div>', unsafe_allow_html=True)

    display_prediction_db()

if __name__ == '__main__':
    init_db()
    show_predict_page()
