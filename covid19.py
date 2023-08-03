
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Function to fetch data from the COVID-19 API
def get_covid_data():
    url = "https://disease.sh/v3/covid-19/countries"
    try:
        response = requests.get(url)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error("Error fetching data from the API:", e)
        return None

# Function for data pre-processing and cleaning
def preprocess_data(covid_data):
    covid_df = pd.DataFrame(covid_data)

    # Keep only relevant columns
    columns_to_keep = ['country', 'cases', 'deaths', 'recovered']
    covid_df = covid_df[columns_to_keep].dropna().drop_duplicates()

    # Normalize and scale numerical features
    numerical_features = ['cases', 'deaths', 'recovered']
    scaler = StandardScaler()
    covid_df[numerical_features] = scaler.fit_transform(covid_df[numerical_features])

    return covid_df

# Function for Linear Regression modeling and prediction
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_linear_regression(model, X_test):
    return model.predict(X_test)

def main():
    st.title('COVID-19 Prediction using Linear Regression')

    # Fetch COVID-19 data from the API
    data = get_covid_data()

    if data:
        # Preprocess and clean the data
        covid_df = preprocess_data(data)

        # Display the preprocessed data
        st.subheader('COVID-19 Data')
        st.dataframe(covid_df)

        # Perform EDA with Plotly
        st.subheader('Exploratory Data Analysis (EDA)')
        fig = px.scatter_matrix(covid_df, dimensions=['cases', 'deaths', 'recovered'], color='country')
        st.plotly_chart(fig)

        # Prepare data for modeling
        X = covid_df[['deaths', 'recovered']]
        y = covid_df['cases']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        model = train_linear_regression(X_train, y_train)

        # Make predictions on the test set
        y_pred = predict_linear_regression(model, X_test)

        # Calculate mean squared error (MSE) as a model evaluation metric
        mse = mean_squared_error(y_test, y_pred)

        # Display model evaluation results
        st.subheader('Model Evaluation')
        st.write(f'Mean Squared Error (MSE): {mse:.2f}')

        # User input for prediction
        st.subheader('COVID-19 Prediction')
        st.write('Enter the number of days to predict future cases:')
        num_days_to_predict = st.number_input('Number of days:', value=7, min_value=1, max_value=30, step=1)

        if st.button('Predict'):
            # Prepare data for future prediction
            future_features = X_test.tail(1).values
            future_predictions = []
            for _ in range(num_days_to_predict):
                future_prediction = predict_linear_regression(model, future_features)
                future_features = [[future_prediction, future_prediction]]
                future_predictions.append(future_prediction[0])

            # Create a DataFrame to display the future predictions
            future_dates = pd.date_range(start=covid_df.index[-1], periods=num_days_to_predict+1)[1:]
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Cases': future_predictions})
            forecast_df.set_index('Date', inplace=True)

            # Display the future predictions
            st.subheader('COVID-19 Cases Forecast')
            st.dataframe(forecast_df)

if __name__ == '__main__':
    main()
