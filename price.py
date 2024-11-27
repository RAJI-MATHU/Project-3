import streamlit as st
import pandas as pd
import numpy as np
import pickle



# Load the saved model, scaler, encoder, and sample data
with open(r'C:\Users\mraja\Downloads\Streamlit\cardekho\Scripts\car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'C:\Users\mraja\Downloads\Streamlit\cardekho\Scripts\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open(r'C:\Users\mraja\Downloads\Streamlit\cardekho\Scripts\encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

#---------------------------------------------------------------------------
car = pd.read_excel(r'C:\Users\mraja\Downloads\Streamlit\cardekho\Scripts\stream_capped_data.xlsx')

#car = pd.read_excel(r'C:\Users\mraja\Downloads\Streamlit\cardekho\Scripts\Cleaned_Cars_Data_Capped_Outlierss.xlsx')
#----------------------------------------------------------------------------------

important_numerical_cols = ['Width','YearofManufacture','kmsDriven', 'Length','Torque', 'Height', 'Mileage','GearBox',
                            'No_owner', 'SeatingCapacity', 'Topspeed','EngineDisplacement']

important_categorical_cols = ['City', 'fueltype', 'bodytype','oemodel', 'Color','Transmission',  'Steering Type','InsuranceValidity']

st.markdown(
    """
    <style>
    .main-heading {
        font-size: 60px;
        color: skyblue;
        text-align: center;
    }
    .subhead{
        font-size: 30px;
        color: #fa8231;
        text-align: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply custom background
background_css = '''
    <style>
    .stApp {
        background-image:url("https://png.pngtree.com/background/20230517/original/pngtree-car-is-shown-on-a-dark-background-with-flames-picture-image_2639919.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        opacity: 0.8; /* Adjust this value to control transparency */
    }
    </style>
'''
#st.markdown(background_css, unsafe_allow_html=True)


# Streamlit UI
st.markdown("<h1 class='main-heading'>Car Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subhead'>Enter the car features below and get an predicted price</h3>", unsafe_allow_html=True)



col1, col2 = st.columns(2)

with col1:
    # Dynamic filtering based on car model
    def filter_data_by_oemodel(car, oemodel):
        return car[car['oemodel'] == oemodel]
    
    oemodel = st.selectbox("Car model", car['oemodel'].unique().tolist())
    
    # Filter data dynamically based on manufacturer
    filtered_data = filter_data_by_oemodel(car, oemodel)
    
    # Update options based on filtered data
    cities = filtered_data['City'].unique().tolist()
    fuel_types = filtered_data['fueltype'].unique().tolist()
    body_types = filtered_data['bodytype'].unique().tolist()
    colors = filtered_data['Color'].unique().tolist()
    transmission_types = filtered_data['Transmission'].unique().tolist()
    steering_types = filtered_data['Steering Type'].unique().tolist()
    Insurance_Validity = filtered_data['InsuranceValidity'].unique().tolist()


   #Categorical
    city = st.selectbox("City", cities,index=None)
    fuel_type = st.selectbox("Fuel Type", fuel_types,index=None)
    body_type = st.selectbox("Body Type", body_types)
    color = st.selectbox("Color", colors,index=None)
    steering_type = st.selectbox("Steering Type", steering_types)
    transmission_type = st.selectbox("Transmission Type", transmission_types)
    Insurance_validity = st.selectbox("Insurance Validity", Insurance_Validity)

    
    
with col2:
    # Numerical inputs based on filtered data
    numerical_ranges = {col: (filtered_data[col].min(), filtered_data[col].max()) for col in important_numerical_cols}

    # Collect numerical inputs
    width = st.number_input("Width", min_value=numerical_ranges['Width'][0], max_value=numerical_ranges['Width'][1], value=numerical_ranges['Width'][0])
    manufacture_year = st.slider("YearofManufacture", min_value=int(numerical_ranges['YearofManufacture'][0]), max_value=int(numerical_ranges['YearofManufacture'][1]), value=int(numerical_ranges['YearofManufacture'][0]))
    kilometers_driven = st.slider("kmsDriven", min_value=numerical_ranges['kmsDriven'][0], max_value=numerical_ranges['kmsDriven'][1], value=numerical_ranges['kmsDriven'][0])
    gear=st.number_input("Seats", min_value=numerical_ranges['GearBox'][0], max_value=numerical_ranges['GearBox'][1], value=numerical_ranges['GearBox'][0])
    torque = st.number_input("Torque", min_value=numerical_ranges['Torque'][0], max_value=numerical_ranges['Torque'][1], value=numerical_ranges['Torque'][0])
    mileage = st.number_input("Mileage", min_value=numerical_ranges['Mileage'][0], max_value=numerical_ranges['Mileage'][1], value=numerical_ranges['Mileage'][0])
    previous_owners = st.selectbox("No_owner", options=[1,2,3,4,5])
    Topspeed=st.slider("Topspeed", min_value=numerical_ranges['Topspeed'][0], max_value=numerical_ranges['Topspeed'][1], value=numerical_ranges['Topspeed'][0])
    
    #previous_owners = st.slider("No_owner", min_value=numerical_ranges['No_owner'][0], max_value=numerical_ranges['No_owner'][1], value=numerical_ranges['No_owner'][0])
    #seats = st.selectbox("Seats", options=range(int(numerical_ranges['SeatingCapacity'][0]), int(numerical_ranges['SeatingCapacity'][1])+1))
    #length = st.number_input("Length", min_value=numerical_ranges['Length'][0], max_value=numerical_ranges['Length'][1], value=numerical_ranges['Length'][0])
    #height = st.number_input("Height", min_value=numerical_ranges['Height'][0], max_value=numerical_ranges['Height'][1], value=numerical_ranges['Height'][0])
    #seats=st.slider("Seats",4,8)
    #EngineDisplacement=st.slider("EngineDisplacement", min_value=numerical_ranges['EngineDisplacement'][0], max_value=numerical_ranges['EngineDisplacement'][1], value=numerical_ranges['EngineDisplacement'][0])



# Collect the numerical and categorical data
numerical_data = np.array([[width, manufacture_year, kilometers_driven, mileage, mileage,gear, torque, mileage, previous_owners,
                            mileage,Topspeed,mileage,gear,gear]])

categorical_data = pd.DataFrame([[city, fuel_type, body_type,oemodel, color,transmission_type,steering_type,Insurance_validity]], 
                                columns=important_categorical_cols)


# Encode categorical data
encoded_data = encoder.transform(categorical_data).toarray()  # Convert sparse matrix to dense array

scaled_data = scaler.transform(numerical_data)

# Ensure both arrays are 2D
numerical_data = np.atleast_2d(numerical_data)

# Check the shapes of both arrays
#st.write(f"Scaled Data Shape: {scaled_data.shape}")
#st.write(f"Encoded Data Shape: {encoded_data.shape}")


# Combine numerical and categorical data
final_data = np.hstack((scaled_data, encoded_data))

# Check the shape and content of final_data
# st.write(f"Final Data Shape: {final_data.shape}")

# Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(final_data)
        st.success(f" The Predicted Car Price is ₹{prediction[0]:,.2f}Lakh")
    except Exception as e:
        st.error(f"Error during prediction: {e}")



