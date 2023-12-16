import streamlit as st
import pandas as pd
import pickle
import json

# Load the columns.json file to get the location names
with open("columns.json", "r") as f:
    columns_data = json.load(f)
    location_names = columns_data['data_columns'][3:]

# Load the pickled Linear Regression model
with open("Banglore_HPP_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

def predict_price(total_sqft, bath, bhk, location_columns):
    # In a real application, you would use your trained model to make predictions here
    # Assuming the model was trained using the same features, create a dummy DataFrame with those features
    input_data = pd.DataFrame({
        'total_sqft': [total_sqft],
        'bath': [bath],
        'bhk': [bhk],
        **{col: [1 if col in location_columns else 0] for col in location_names}
    })

    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.title("House Price Prediction App")

    # Add input fields in the sidebar
    st.sidebar.header("Input Parameters")
    total_sqft = st.sidebar.number_input("Total Sqft", min_value=100, max_value=10000, step=100)
    bath = st.sidebar.number_input("Bath", min_value=1, max_value=10, step=1)
    bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, step=1)
    selected_location = st.sidebar.selectbox("Select Location", location_names)

    # Display user inputs
    st.write("### User Input:")
    st.write(f"Total Sqft: {total_sqft}")
    st.write(f"Bath: {bath}")
    st.write(f"BHK: {bhk}")
    st.write(f"Selected Location: {selected_location}")

    # Add a "Predict" button
    if st.button("Predict"):
        # Make a prediction based on user inputs
        prediction = predict_price(total_sqft, bath, bhk, [selected_location])

        # Display the predicted price
        st.write("### Predicted Price:")
        st.write(f"${prediction:,.2f}")

if __name__ == "__main__":
    main()
