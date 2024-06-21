import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Function to load data with caching
@st.cache_data
def load_data():
    data_path = 'projectfin.csv'
    if not os.path.exists(data_path):
        st.error(f"The file {data_path} does not exist.")
        return None
    return pd.read_csv(data_path)

# Function to preprocess data and train model
@st.cache_data
def preprocess_and_train(data):
    if data is None:
        return None, None, None

    # Select features and target variable
    features = data[['Overall_Rating', 'Food_Rating', 'Security_Rating', 'Hospitality_Rating', 'Room_Sharing', 'Lnum', 'Areanum', 'prioritypoints']]
    target = data['Rent_(monthly)']

    # Preprocessor to scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), features.columns)]
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Define the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2

# Function to make predictions
def predict(model, input_data):
    input_df = pd.DataFrame([input_data])
    input_df = model.named_steps['preprocessor'].transform(input_df)
    prediction = model.named_steps['regressor'].predict(input_df)
    return prediction[0]

# Load data
data = load_data()
if data is not None:
    # Preprocess data and train model
    model, mae, r2 = preprocess_and_train(data)

    # Create a mapping dictionary for area names to numbers
    area_mapping = {
        "Select your area": 0,
        "Mehidipatnam": 1,
        "Gachibowli": 2,
        "Kukatpally": 3,
        "Madhapur": 4,
        "Lakdikapul": 5,
        "Narsingi": 6,
        "Panjagutta": 7,
        "Kondapur": 8, 
        "Bandlaguda Jagir": 9,
        "Moinabad": 10,
        "Shamshabad": 11, 
        "Attapur": 12,
        "Budwel": 13,
        "Kismatpur": 14,
        "Nanalnagar": 15,
        "Himayatnagar": 16,
        "Langerhouz": 17,
        "Miyapur": 18,
        "Amberpet": 19,
        "LB Nagar": 20,
        "Secunderabad": 21,
        "Suncity": 22,
        "Kalimandir": 23,
        "Aziznagar": 24,
        "Rajendranagar": 25
    }

    # Streamlit UI
    st.title("Hostel Recommendation System")
    st.write("Your ideal Hostel Experience Starts here...!!!")
    st.write("Find Your perfect hostel across the city in your hands.....")
    st.write("select your preferences and explore now...")

    selected_area = st.selectbox('Area', list(area_mapping.keys()))
    overall_rating = st.slider("Overall Rating", min_value=0.0, max_value=5.0, value=2.5, step=0.5)
    food_rating = st.slider("Food Rating", min_value=0.0, max_value=5.0, value=2.5, step=0.5)
    security_rating = st.slider("Security Rating", min_value=0.0, max_value=5.0, value=2.5, step=0.5)
    hospitality_rating = st.slider("Hospitality Rating", min_value=0.0, max_value=5.0, value=2.5, step=0.5)
    room_sharing = st.slider("Room Sharing", min_value=0, max_value=8, value=5)
    laundry_option = st.selectbox("Laundry", options=[("No", 0), ("Yes", 1)])
    laundry = laundry_option[1]

    if st.button("Predict"):
        selected_row = data[data['Areanum'] == area_mapping[selected_area]].iloc[0]
        priority_points = selected_row['prioritypoints']

        input_data = {
            'Areanum': area_mapping[selected_area],
            'Overall_Rating': overall_rating,
            'Food_Rating': food_rating,
            'Security_Rating': security_rating,
            'Hospitality_Rating': hospitality_rating,
            'Room_Sharing': room_sharing,
            'Lnum': laundry,
            'prioritypoints': priority_points
        }

        # Make prediction
        prediction = predict(model, input_data)
        st.write(f'Predicted Rent: {prediction}')

        # Filter data based on the selected area and the predicted rent range
        predicted_rent = prediction
        rent_lower_bound = max(0, predicted_rent - 700)
        rent_upper_bound = predicted_rent + 700

        filtered_data = data[(data['Areanum'] == area_mapping[selected_area]) & 
                             (data['Rent_(monthly)'] >= rent_lower_bound) & 
                             (data['Rent_(monthly)'] <= rent_upper_bound)]

        # Filter data based on the selected area and the predicted rent range
        #filtered_data = data[(data['Areanum'] == area_mapping[selected_area]) & 
                             #(data['Rent_(monthly)'] >= prediction - 100) & 
                             #(data['Rent_(monthly)'] <= prediction + 100)]

        st.write("Hostels with similar rent and ratings in the selected area:")
        st.write(filtered_data[['Hostel_Name', 'Rent_(monthly)', 'Room_Sharing', 'Nearest_Bus_Stop_Dist', 'Price_(per day)', 'Other_Amenities','Overall_Rating']])

        st.write(f'Mean Absolute Error: {mae}')
        st.write(f'R2 Score: {r2}')
else:
    st.write("Unable to load data. Please check the file path.")
