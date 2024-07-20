import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load the trained model
with open('pipe.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Function to make predictions
def predict_winner(input_data):
    prediction_proba = model.predict_proba(input_data)
    return prediction_proba 


# Define the app
st.title("IPL Win Predictor")
    
       
# Input features for the prediction
st.header("Input Features")
    
batting_team = st.selectbox("🏏Batting Team", ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Delhi Capitals',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals'])
bowling_team = st.selectbox("🏀Bowling Team", ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Delhi Capitals',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals'])
city = st.selectbox("Select Venue", ['Bangalore', 'Delhi', 'Hyderabad', 'Chennai', 'East London',
       'Mumbai', 'Abu Dhabi', 'Cape Town', 'Chandigarh', 'Durban',
       'Kolkata', 'Centurion', 'Jaipur', 'Cuttack', 'Raipur', 'Sharjah',
       'Dharamsala', 'Johannesburg', 'Nagpur', 'Ahmedabad',
       'Visakhapatnam', 'Kimberley', 'Ranchi', 'Port Elizabeth',
       'Bloemfontein', 'Pune', 'Indore'])

Score = st.number_input('Score', min_value=0, max_value=500, step=1)
    
over = st.number_input('Overs', min_value=0, max_value=120, step=1)
    
wicket = st.number_input('Wickets', min_value=0, max_value=10, step=1)
      
Target = st.number_input('🎯Target', min_value=0, max_value=500, step=1)
    



if st.button('🎲Predict'):
    
    
    Runs_left=Target-Score
    ball_left=120-(over*6)
    wicket_left=10-wicket
    Current_rr=Score/over
    Required_rr=Runs_left/over
        # Prepare input data in the format your model expects
    input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'Runs_left': [Runs_left],
            'ball_left': [ball_left],
            'wicket_left': [wicket_left],
            'Current_rr': [Current_rr],
            'total_runs_y': [Target],
            'Required_rr': [Required_rr]
        })
        
   
    prediction_proba = predict_winner(input_data)
        
    # Display the result
   
    st.write(f"Current Winning🏆 Probability: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Current Losing💀 Probability: {prediction_proba[0][0]*100:.2f}%")
    

        
        
