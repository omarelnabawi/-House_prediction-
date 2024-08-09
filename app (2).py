import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load and preprocess the dataset
df = pd.read_csv('housing.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Initialize the OrdinalEncoder
en = OrdinalEncoder()

# Select and encode object-type columns
df_obj = df.select_dtypes(include=['object'])
mappings = {}
for column in df_obj.columns:
    df[column] = en.fit_transform(df[[column]])
    categories = en.categories_[0]
    encoded_values = list(range(len(categories)))
    column_mapping = dict(zip(categories, encoded_values))
    mappings[column] = column_mapping

# Split the dataset into features and target variable
x = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# Load the pre-trained model (ensure that 'xg model.pkl' exists)
model = joblib.load('xg.pkl')

# Calculate predictions and R² scores
y_pred = model.predict(x_test)
y_tpred = model.predict(x_train)
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_tpred)

# Streamlit app
st.title("House Price Prediction")

# Display R² scores
st.write(f"R-squared (Train): {r2_train:.2f}")
st.write(f"R-squared (Test): {r2_test:.2f}")

# Collect user input for prediction
st.subheader("Enter the feature values:")
user_input = {}
for col in x.columns:
    if col in mappings:
        user_input[col] = st.selectbox(f"{col}", options=list(mappings[col].keys()))
    else:
        user_input[col] = st.number_input(f"{col}", min_value=float(x[col].min()), max_value=float(x[col].max()), step=0.01)

# Process the input
input_df = pd.DataFrame([user_input])
for col in input_df.columns:
    if col in mappings:
        input_df[col] = input_df[col].map(mappings[col])

# Ensure the input data is in numeric format
input_df = input_df.apply(pd.to_numeric)

# Predict and display the result
if st.button("Submit"):
    prediction = model.predict(input_df)
    st.header(f"Prediction: {prediction[0]:.2f}")

