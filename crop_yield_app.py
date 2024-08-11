import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import datetime

# Load the trained model pipeline
model_pipeline = joblib.load('crop_yield_model.pkl')

# Load the dataset for visualization
data = pd.read_csv('crop_yield.csv')

# Define a function for prediction
def predict_yield(data):
    return model_pipeline.predict(data)

# Define a function for plotting crop yield over time
def plot_yield_over_time(crop):
    filtered_data = data[data['Crop'] == crop]
    if filtered_data.empty:
        return None
    yield_over_time = filtered_data.groupby('Crop_Year')['Yield'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=yield_over_time, x='Crop_Year', y='Yield', marker='o')
    plt.title(f'Average Yield of {crop} Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Yield')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

# Define a function for displaying customized solutions
def display_solutions(crop, state, season):
    solutions = {
        'Rice': {
            'Kharif': {
                'General': "Rice cultivation during the Kharif season benefits from the monsoon rains, reducing the need for irrigation. Consider using high-yield and pest-resistant varieties.",
                'Andhra Pradesh': "In Andhra Pradesh, focus on System of Rice Intensification (SRI) techniques, which involve alternate wetting and drying, to optimize water use and increase yield.",
                
            },
            'Rabi': {
                'General': "During the Rabi season, water availability is a critical issue for rice. Opt for short-duration, drought-resistant varieties and consider supplemental irrigation.",
                'Punjab': "In Punjab, using laser leveling can help improve water distribution efficiency, reducing the water requirement for Rabi rice.",
                
            }
        },
        'Wheat': {
            'Rabi': {
                'General': "Wheat requires a cool climate and adequate irrigation. Use certified seeds of high-yielding, disease-resistant varieties. Apply appropriate doses of nitrogen, phosphorus, and potassium fertilizers.",
                'Punjab': "In Punjab, adopt zero-tillage techniques to conserve soil moisture and reduce sowing time. Use varieties like PBW-550 for higher yield.",
                
            }
        },
        'Maize': {
            'Kharif': {
                'General': "For Kharif maize, use hybrid varieties that can withstand high rainfall and resist pests.",
                'Karnataka': "In Karnataka, maize farmers should adopt integrated nutrient management practices and ensure timely sowing for optimal growth.",
                
            },
            'Rabi': {
                'General': "For Rabi maize, ensure adequate irrigation and choose varieties suitable for cooler temperatures. Crop rotation with legumes can enhance soil fertility.",
                'Bihar': "In Bihar, use high-yielding hybrids like Ganga-11 and optimize fertilizer use to maximize yield.",
               
            }
        }
        
    }

    general_solution = solutions.get(crop, {}).get(season, {}).get('General', "No general solution available for this combination.")
    specific_solution = solutions.get(crop, {}).get(season, {}).get(state, "No region-specific solution available for this combination.")

    st.write(f"### General Solutions for {crop} ({season})")
    st.markdown(general_solution)

    st.write(f"### Region-Specific Solutions for {crop} in {state} ({season})")
    st.markdown(specific_solution)

# Streamlit app layout
st.title('Comprehensive Crop Yield Prediction and Solutions')

# Input fields for features
crop = st.selectbox('Select Crop', [
    "Arecanut", "Arhar/Tur", "Bajra", "Banana", "Barley", "Black pepper",
    "Cardamom", "Cashewnut", "Castor seed", "Coconut", "Coriander",
    "Cotton(lint)", "Cowpea(Lobia)", "Dry chillies", "Garlic", "Ginger",
    "Gram", "Groundnut", "Guar seed", "Horse-gram", "Jowar", "Jute",
    "Khesari", "Linseed", "Maize", "Masoor", "Mesta", "Moong(Green Gram)",
    "Moth", "Niger seed", "Oilseeds total", "Onion", "Other Rabi pulses",
    "Other Cereals", "Other Kharif pulses", "Other oilseeds",
    "Other Summer Pulses", "Peas & beans (Pulses)", "Potato", "Ragi",
    "Rapeseed & Mustard", "Rice", "Safflower", "Sannhamp", "Sesamum",
    "Small millets", "Soyabean", "Sugarcane", "Sunflower", "Sweet potato",
    "Tapioca", "Tobacco", "Turmeric", "Urad", "Wheat"
])
crop_year = st.number_input('Crop Year', min_value=1997, max_value=2040, step=1)
season = st.selectbox('Season', ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter', 'Autumn'])
state = st.selectbox('State', [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Puducherry", "Punjab", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
])
area = st.number_input('Area (in hectares)', min_value=0.0, value=100.0)
production = st.number_input('Production (in tons)', min_value=0.0, value=100.0)
annual_rainfall = st.number_input('Annual Rainfall (in mm)', min_value=0.0, value=1000.0)
fertilizer = st.number_input('Fertilizer (in kg/ha)', min_value=0.0, value=1000.0)
pesticide = st.number_input('Pesticide (in kg/ha)', min_value=0.0, value=10.0)

# Additional features: Weather data input
st.sidebar.header('Additional Data Inputs')
weather_temp = st.sidebar.number_input('Current Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0)
weather_humidity = st.sidebar.number_input('Current Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
weather_wind_speed = st.sidebar.number_input('Wind Speed (km/h)', min_value=0.0, max_value=200.0, value=10.0)
weather_date = st.sidebar.date_input('Date', datetime.date.today())

# Predict yield
if st.button('Predict'):
    # Create input data frame
    input_data = pd.DataFrame({
        'Crop': [crop], 'Crop_Year': [crop_year], 'Season': [season], 
        'State': [state], 'Area': [area], 'Production': [production],
        'Annual_Rainfall': [annual_rainfall], 'Fertilizer': [fertilizer],
        'Pesticide': [pesticide]
    })
    
    # Predict yield
    try:
        prediction = predict_yield(input_data)
        st.success(f'Estimated Crop Yield: {prediction[0]:.2f} tons/ha')
        st.header('Recommended Solutions')
        # Solution 1: Optimized Resource Allocation
        st.subheader('Optimized Resource Allocation')
        st.write('Based on the predicted yield, consider the following:')
        st.write('- Adjust fertilizer usage to maintain soil health and optimize crop yield.')
        st.write('- Efficient water management practices to conserve water and reduce costs.')
        st.write('- Use integrated pest management to reduce the use of pesticides.')

        # Solution 2: Crop Management Strategies
        st.subheader('Crop Management Strategies')
        st.write('To maximize yield:')
        st.write('- Monitor weather forecasts and adjust planting schedules accordingly.')
        st.write('- Implement crop rotation to improve soil fertility and reduce pest infestations.')

        # Solution 3: Risk Mitigation
        st.subheader('Risk Mitigation')
        st.write('To mitigate risks associated with crop failure:')
        st.write('- Develop contingency plans for adverse weather conditions.')
        st.write('- Invest in crop insurance to protect against financial losses.')

        # Solution 4: Policy Recommendations
        st.subheader('Policy Recommendations')
        st.write('Policy makers can:')
        st.write('- Provide subsidies for sustainable farming practices.')
        st.write('- Offer training programs to educate farmers on efficient resource use.')
        st.write('- Support infrastructure development for better irrigation and storage facilities.')

    except ValueError as e:
        st.error(f"Error: {e}")

# Visualization section
st.subheader('Crop Yield Visualization Over Time')
# Plot yield over time
yield_plot = plot_yield_over_time(crop)
if yield_plot:
    st.image(yield_plot, caption=f'Average Yield of {crop} Over Time')
else:
    st.warning('No data available for the selected crop.')

# Solutions section
st.subheader('Customized Solutions for Improving Crop Yield')
display_solutions(crop, state, season)

# Comparative Analysis
st.subheader('Comparative Yield Analysis')
comparison_crop = st.selectbox('Select another Crop for Comparison', [
    "Arecanut", "Arhar/Tur", "Bajra", "Banana", "Barley", "Black pepper",
    "Cardamom", "Cashewnut", "Castor seed", "Coconut", "Coriander",
    "Cotton(lint)", "Cowpea(Lobia)", "Dry chillies", "Garlic", "Ginger",
    "Gram", "Groundnut", "Guar seed", "Horse-gram", "Jowar", "Jute",
    "Khesari", "Linseed", "Maize", "Masoor", "Mesta", "Moong(Green Gram)",
    "Moth", "Niger seed", "Oilseeds total", "Onion", "Other Rabi pulses",
    "Other Cereals", "Other Kharif pulses", "Other oilseeds",
    "Other Summer Pulses", "Peas & beans (Pulses)", "Potato", "Ragi",
    "Rapeseed & Mustard", "Rice", "Safflower", "Sannhamp", "Sesamum",
    "Small millets", "Soyabean", "Sugarcane", "Sunflower", "Sweet potato",
    "Tapioca", "Tobacco", "Turmeric", "Urad", "Wheat"
])
if st.button('Compare Yields'):
    comparison_plot = plot_yield_over_time(comparison_crop)
    if comparison_plot:
        st.image(comparison_plot, caption=f'Average Yield of {comparison_crop} Over Time')
    else:
        st.warning('No data available for the selected comparison crop.')

# Resource Recommendations
st.subheader('Resource Recommendations')
st.markdown("""
- [Government Schemes](https://pib.gov.in/PressReleaseIframePage.aspx?PRID=2002012): Learn about government schemes for farmers.
- [Best Practices](https://upagripardarshi.gov.in/Index.aspx): Read about best agricultural practices.
- [Weather Forecast](https://www.accuweather.com/): Get the latest weather forecast.
- [Market Prices](https://agmarknet.gov.in/PriceAndArrivals/CommodityDailyStateWise.aspx): Check the latest market prices for crops.
""")

# Feedback Section
st.subheader('Provide Feedback')
feedback = st.text_area("Your Feedback", "")
if st.button('Submit Feedback'):
    st.success("Thank you for your feedback!")
    # Here, you could also save the feedback to a file or database
