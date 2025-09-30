import streamlit as st
import numpy as np
# import joblib
import pandas as pd
import pickle
import os

st.title("Bank Marketing Campaign Prediction")
st.write("Predicting Wheter Customer Will Subscribe to Term Deposit")

user_session = st.session_state

if 'y' not in user_session:
    user_session['y'] = 0

##testing markdown
st.markdown(f"""
    # Prediction Result: 
    # :red[{user_session['y']}]
""")

# --- User Input Section ---
st.header("Client Information")
st.markdown("Please provide the following details to get a prediction.")

# Define categorical options based on original data
job_options = ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student']
marital_options = ['married', 'single', 'divorced']
education_options = ['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
contact_options = ['unknown', 'cellular', 'telephone']
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_of_week_options = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
poutcome_options = ['unknown', 'failure', 'other', 'success']

# Create input widgets for each feature
age = st.slider("Age", 18, 100, 30)
job = st.selectbox("Job", options=job_options)
marital = st.selectbox("Marital Status", options=marital_options)
education = st.selectbox("Education", options=education_options)
default = st.radio("Has Credit Default?", options=['yes', 'no'])
housing = st.radio("Has a Housing Loan?", options=['yes', 'no'])
loan = st.radio("Has a Personal Loan?", options=['yes', 'no'])
contact = st.selectbox("Contact Communication Type", options=contact_options)
day_of_week = st.selectbox("Last Contact Day of Month", options=day_of_week_options)
month = st.selectbox("Last Contact Month of Year", options=month_options)
duration = st.number_input("Last Contact Duration (in seconds)", min_value=0, max_value=6000, value=150)
campaign = st.slider("Number of Contacts During This Campaign", 1, 60, 2)
pdays = st.slider("Days Since Previous Contact", -1, 900, -1)
previous = st.slider("Number of Contacts Before This Campaign", 0, 30, 0)
poutcome = st.selectbox("Outcome of Previous Campaign", options=poutcome_options)
emp_var_rate = st.slider("(Macroeconomic) emp.var.rate", -3.5, 1.5, 0.1)
cons_price_idx = st.slider("(Macroeconomic) cons.price.idx", 92.000, 95.000, 0.001)
cons_conf_idx = st.slider("(Macroeconomic) cons.conf.idx", -60.1, -25.1, 0.1)
euribor3m = st.slider("(Macroeconomic) euribor3m", 0.612, 0.551, 0.001)
nr_employed = st.slider("(Macroeconomic) nr.employed", 4900.5, 5510.1, 0.1)
past_contacted = st.radio("Previously contacted?", options=[1, 0])

################################################################# TEST
            
            
if education == 'unknown' :
    education = 1
elif education == 'illiterate' :
    education = 1
elif education == 'basic.4y' :
    education = 2
elif education == 'basic.6y' :
    education = 3
elif education == 'basic.9y' :
    education = 4
elif education == 'high.school' :
    education = 5
elif education == 'professional.course' :
    education = 6
elif education == 'university.degree' :
    education = 7
else:
    education = 1

###

if month == 'jan' :
    month = 1
elif month == 'feb' :
    month = 2
elif month == 'mar' :
    month = 3
elif month == 'apr' :
    month = 4
elif month == 'may' :
    month = 5
elif month == 'jun' :
    month = 6
elif month == 'jul' :
    month = 7
elif month == 'aug' :
    month = 8
elif month == 'sep' :
    month = 9
elif month == 'oct' :
    month = 10
elif month == 'nov' :
    month = 11
else:
    month = 12
####

if day_of_week == 'mon' :
    day_of_week = 1
elif day_of_week == 'tue' :
    day_of_week = 2
elif day_of_week == 'wed' :
    day_of_week = 3
elif day_of_week == 'thu' :
    day_of_week = 4
elif day_of_week == 'fri' :
    day_of_week = 5
elif day_of_week == 'sat' :
    day_of_week = 6
else:
    day_of_week = 7

#################################################################


user_data = pd.DataFrame([[
                age, job, marital, education, default, housing, loan, contact,
                month, day_of_week, duration, campaign, pdays, previous, poutcome, 
                emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, past_contacted
            ]], columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                          'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome',
                          'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'past_contacted'])
################################################################
# # Test With Pickle : separated file for preprocessing and model
# #import encoder
# with open("preproc_enigma.pkl", 'rb') as f:
#     data_encoder = pickle.load(f)
#     # data_encoder = joblib.load(f)

# #encode inputs
# X_pred = data_encoder.transform(user_data)

# #import model
# with open("model_enigma.pkl", "rb") as f:
#     model = pickle.load(f)

# y_pred = model.predict(X_pred)
# user_session['y'] = y_pred[0]

###############################################################
# Test With Pickle : one pickle pipeline

#import pipeline
with open("pipeline_enigma.pkl", "rb") as f:
    pipeline = pickle.load(f)

y_pred = pipeline.predict(user_data)
user_session['y'] = y_pred[0]


################################################################
# # Test With joblib : one joblib pipeline

# # X_pred = data_encoder.transform(user_data)
# loaded_pipeline = joblib.load('pipeline_finpro_enigma.joblib ')
# y_pred = loaded_pipeline.predict(user_data)

# user_session['y'] = y_pred[0]

################################################################

if st.button("Submit"):
    user_session['y']+=1