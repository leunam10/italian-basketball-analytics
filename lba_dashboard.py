
"""

This script is used to create a streamlit dashboard to show the hystorical and present
data of the teams and players from 2003-2004 to 2023-2024 seasons.

"""



import pandas as pd
import streamlit as st
import sys
import os


path = os.path.dirname(os.path.abspath("lba_dashboard.py"))
data_path = os.path.join(path, "data")

@st.cache_data
def read_data(dataset):
    if(dataset == "teams"):
        ## teams
        df = pd.read_csv(os.path.join(data_path, "teams_stats_2003-2004_2023-2024.csv"))
    elif(dataset == "players"):
        ## players
        df = pd.read_csv(os.path.join(data_path, "players_stats_2003-2004_2023-2024.csv"))
        
    return df
        
if(__name__ == "__main__"):

    # read csv files
    teams_df = read_data("teams")
    player_df =read_data("players")
    
    # streamlit dashboard
    
    ## setup the title
    st.title("🏀 Italian Basketball A League Analytics")
    
    ## data selection 
    st.header("Data Selection")
    data_choice = st.radio("Select the data you want to analyze",
                           ["LBA Teams", "LBA Players"])
    
    if(data_choice == "LBA Teams"):
        start_year = teams_df["Year"].min()
        end_year = teams_df["Year"].max()
    elif(data_choice == "LBA Players"):
        start_year = players_df["Year"].min()
        end_year = players_df["Year"].max()

        
    season_interval = st.slider("Select the range of years:",
                                  min_value = start_year, max_value = end_year,
                                  value = (start_year, end_year)) 

    st.write("You have selected the seasons from", season_interval[0], "to", season_interval[1])
    
