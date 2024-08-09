# House Price Prediction App

This repository contains a Streamlit web application for predicting house prices based on various features. The model is trained using a dataset of housing data and utilizes a pre-trained model to make predictions based on user input.

## Features

- **R² Score Display**: Shows the R² score for both training and test datasets to evaluate model performance.
- **User Input**: Allows users to input values for different features to predict the house price.
- **Prediction Output**: Displays the predicted house price based on user input.

## Installation

To run this application, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
2. **Navigate to the project directory**:
   ```bash
     cd house-price-prediction
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
# Usage
1.Launch the application using the command above.
2.Input the values for each feature in the sidebar.
3.Click the "Submit" button to see the predicted house price.
# Files
- **app.py**:The main application file containing the Streamlit code.
- **sample_data.csv**:Sample dataset used for training , testing and anylitics the ocean proximity feature.
- **xg model.pkl**: Pre-trained model file (ensure it exists in your working directory).
# Model Details
- **Model Used**: XGBRegressor (pre-trained and saved using joblib)
- **Metrics**:
            - R² Score
# License
This project is licensed under the MIT License - see the LICENSE file for details.
# Acknowledgments
- The dataset used in this project is sourced from Kaggle
- Thanks to Bharat for their guidance and support.
