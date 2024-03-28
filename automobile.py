import pandas as pd
import joblib
import streamlit as st 
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('automobile (1).csv')

st.markdown("<h1 style = 'color: #5F5D9C; text-align: center; font-family: helvetica'>STARTUP PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By ZAtAkerele Data Science Cohort</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)

st.image('pngwing.com(2).png')
st.header('Project Info', divider = True)
st.write("The primary objective of this machine learning project is to develop a predictive model capable of assessing the likelihood of customer loan approval based on a comprehensive set of features and criteria. By harnessing the power of advanced ML techniques, we aim to provide lending institutions with a reliable tool for automating and optimizing the loan approval process. The ultimate goal is to enhance decision-making efficiency, mitigate risks, and improve customer satisfaction by delivering faster and more accurate loan approvals")

st.dataframe(data)
# import the transformers 
make_enc = joblib.load('make_encoder.pkl')
body_enc = joblib.load('body-style_encoder.pkl')

st.sidebar.image('pngwing.com(3).png')
st.sidebar.divider()
curb = st.sidebar.number_input('CURB WEIGHT', data['curb-weight'].min(), data['curb-weight'].max())
symboling = st.sidebar.number_input('SYMBOLING', data['symboling'].min(), data['symboling'].max())
make = st.sidebar.selectbox('CAR MAKE', data['make'].unique())
horse = st.sidebar.number_input('HORSE POWER')
engine = st.sidebar.number_input('ENGINE SIZE', data['engine-size'].min(), data['engine-size'].max())
body = st.sidebar.selectbox('BODY STYLE', data['body-style'].unique())
wheel = st.sidebar.number_input('WHEEL BASE', data['wheel-base'].min(), data['wheel-base'].max())
city = st.sidebar.number_input('CITY MPG', data['city-mpg'].min(), data['city-mpg'].max())

input_vars = pd.DataFrame()
input_vars['curb-weight'] = [curb]
input_vars['symboling'] = [symboling]
input_vars['make'] = [make]
input_vars['horsepower'] = [horse]
input_vars['engine-size'] = [engine]
input_vars['body-style'] = [body]
input_vars['wheel-base'] = [wheel]
input_vars['city-mpg'] = [city]

st.subheader('Input Variable', divider = True)
st.dataframe(input_vars)

input_vars['make'] = make_enc.transform(input_vars['make'])
input_vars['body-style'] = body_enc.transform(input_vars['body-style'])

model = joblib.load('carPriceModel.pkl')

if st.button('Predict Car Price'):
    predicted = model.predict(input_vars)
    st.success(f'The predicted value of your car is ${int(predicted[0])}')