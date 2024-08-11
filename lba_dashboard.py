
"""

This script is used to create a streamlit dashboard to show the hystorical and present
data of the teams and players from 2003-2004 to 2023-2024 seasons.

"""



import pandas as pd
import streamlit as st
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
    players_df = read_data("players")
    
    # streamlit dashboard

    
    ## setup the title
    st.title("🏀 Italian Basketball A League Analytics")
    
    
    ## data selection 
    st.sidebar.header("Dataset Selection")
    data_choice = st.sidebar.radio("Select the data you want to analyze",
                                  ["LBA Teams", "LBA Players"])
    
    if(data_choice == "LBA Teams"):
        start_year = teams_df["Year"].min()
        end_year = teams_df["Year"].max()
    elif(data_choice == "LBA Players"):
        start_year = players_df["Year"].min()
        end_year = players_df["Year"].max()


    if(data_choice == "LBA Teams"):
        st.header("LBA Teams Dataframe")
        st.dataframe(teams_df)
    elif(data_choice == "LBA Players"):
        st.header("LBA Players Dataframe")
        st.dataframe(players_df)
        
    st.sidebar.header("Time Selection")
    season_interval = st.sidebar.slider("Select the range of years:",
                                        min_value = start_year, max_value = end_year,
                                        value = (start_year, end_year)) 

    st.sidebar.write("You have selected the seasons from", season_interval[0], "to", season_interval[1])

    if(data_choice == "LBA Teams"):
        st.sidebar.header("Teams Selection")
        select_all_toggle = st.sidebar.toggle("Activate to have all the teams selected at once")
        if(select_all_toggle):
            st.sidebar.write(":red[You have selected all the teams in the dataset]")
        else:
            st.sidebar.multiselect("Select the team(s)", list(teams_df["Team"].unique()))

    elif(data_choice == "LBA Players"):
        st.sidebar.header("Players Selection")
        select_all_toggle = st.sidebar.toggle("Activate to have all the players selected at once")
        if(select_all_toggle):
            st.sidebar.markdown(":red[You have selected all the players in the dataset]")
        else:
            st.sidebar.multiselect("Select the team(s)", list(players_df["Player"].unique()))


    # select the dataframe
    if(select_all_toggle):        
        pass
    else:
        pass