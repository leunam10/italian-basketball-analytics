
"""

This script is used to create a streamlit dashboard to show the hystorical and present
data of the teams and players from 2003-2004 to 2023-2024 seasons.

"""



import pandas as pd
import streamlit as st
import plotly.express as px
import os


path = os.path.dirname(os.path.abspath("lba_dashboard.py"))
data_path = os.path.join(path, "data")

@st.cache_data
def read_data(dataset):

    """

    This method allows to read the dataframes that are used in the dashboard.
    With the decorator @st.cache_data the loading is done just the first time
    the dashboard is run.

    - dataset (str):  which dataset to load as pandas dataframe
    
    return
    - df: pandas datafream 

    """

    if(dataset == "teams"):
        ## teams
        df = pd.read_csv(os.path.join(data_path, "teams_stats_2003-2004_2023-2024.csv"))
    elif(dataset == "players"):
        ## players
        df = pd.read_csv(os.path.join(data_path, "players_stats_2003-2004_2023-2024.csv"))
    else:
        print("The selected dataset does not exists. The 'teams' dataset will be loaded")    
        df = pd.read_csv(os.path.join(data_path, "teams_stats_2003-2004_2023-2024.csv"))

    return df
        

def df_selector(df, dataset, dataset_element, season_interval, stat):

    """

    This method allows to select a dataframe with respect the specific player or team
    and for a specific statistic.

    - df: pandas dataframe to select
    - dataset (str): which dataset will be selected
    - dataset_element (dict): the values of the dict are the name of the player or the team
    - season_interval (tuple): the interval of year to consider
    - stat (str): which statistic to select

    return
    - select_df: a pandas dataframe selection of the input dataframe

    """    

    # select by team/player and stat. The Year is selected later
    if(dataset_element != "all"):
        if(dataset == "LBA Teams"):
            if(select_stat == None):
                select_df = df[df["Team"].isin(dataset_element)][["Team", "Playoff", "Finalist", "Winner", "Year"]]
            else:
                select_df = df[df["Team"].isin(dataset_element)][["Team", "Playoff", "Finalist", "Winner", "Year", stat]]
        elif(dataset == "LBA Players"):
            if(select_stat == None):
                select_df = df[df["Player"].isin(dataset_element)][["Player", "Year", "Team", "MVP"]]
            else:
                select_df = df[df["Player"].isin(dataset_element)][["Player", "Year", "Team", "MVP", stat]]
    elif(dataset_element == "all"):
        if(dataset == "LBA Teams"):
            if(select_stat == None):
                select_df = df[["Team", "Playoff", "Finalist", "Winner", "Year"]]
            else:
                select_df = df[["Team", "Playoff", "Finalist", "Winner", "Year", stat]]
        elif(dataset == "LBA Players"):
            if(select_stat == None):
                select_df = df[["Player", "Year", "Team", "MVP"]]
            else:
                select_df = df[["Player", "Year", "Team", "MVP", stat]]

    # select by year
    select_df = select_df.loc[(select_df["Year"]>=season_interval[0]) & (select_df["Year"]<=season_interval[1])]

    return select_df

if(__name__ == "__main__"):

    # read csv files
    teams_df = read_data("teams")
    players_df = read_data("players")
    
    # streamlit dashboard

    ## setup the title
    st.title("🏀 Italian Basketball A League Analytics")
        
    ## dataset selection
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
        
    ## time interval selection
    st.sidebar.header("Time Selection")
    season_interval = st.sidebar.slider("Select the range of years:",
                                        min_value = start_year, max_value = end_year,
                                        value = (start_year, end_year)) 

    st.sidebar.write("You have selected the seasons from", season_interval[0], "to", season_interval[1])

    ## teams or players selection
    if(data_choice == "LBA Teams"):
        st.sidebar.header("Teams Selection")
        select_all_toggle = st.sidebar.toggle("Activate to have all the teams selected at once")
        if(select_all_toggle):
            st.sidebar.write(":red[You have selected all the teams in the dataset]")
            dataset_element = "all"
        else:
            dataset_element = st.sidebar.multiselect("Select the team(s)", list(teams_df["Team"].unique()))

    elif(data_choice == "LBA Players"):
        st.sidebar.header("Players Selection")
        select_all_toggle = st.sidebar.toggle("Activate to have all the players selected at once")
        if(select_all_toggle):
            st.sidebar.markdown(":red[You have selected all the players in the dataset]")
            dataset_element = "all"
        else:
            dataset_element = st.sidebar.multiselect("Select the player(s)", list(players_df["Player"].unique()))

    ## statistics selection
    st.sidebar.header("Statistics Selection")
    if(data_choice == "LBA Teams"):
        teams_cols = teams_df.columns
        cols_to_remove = ["Team", "Year", "Playoff", "Finalist", "Winner"]
        teams_stats = list(filter(lambda x: x not in cols_to_remove, teams_cols))
        select_stat = st.sidebar.selectbox("Select the statistics you want to show",
                                           (teams_stats), index=None,
                                           placeholder="Select the statistic...")
    elif(data_choice == "LBA Players"):
        players_cols = players_df.columns
        cols_to_remove = ["Player", "Year", "Team", "MVP"]
        players_stats = list(filter(lambda x: x not in cols_to_remove, players_cols))
        select_stat = st.sidebar.selectbox("Select the statistics you want to show",
                                           (players_stats), index=None,
                                           placeholder="Select the statistic...")


    ## select the dataframe
    if(select_all_toggle): 
        if(data_choice == "LBA Teams"):
            if(dataset_element == "all"):
                select_df = df_selector(teams_df, data_choice, "all", season_interval, select_stat)
        elif(data_choice == "LBA Players"):
            if(dataset_element == "all"):
                select_df = df_selector(layers_df, data_choice, "all", season_interval, select_stat)
    else:
        if(data_choice == "LBA Teams"):
            if(len(dataset_element)>0):
                select_df = df_selector(teams_df, data_choice, dataset_element, season_interval, select_stat)
        elif(data_choice == "LBA Players"):
            if(len(dataset_element)>0):
                select_df = df_selector(layers_df, data_choice, dataset_element, season_interval, select_stat)


    try:
        fig = px.bar(
                select_df,
                x="Team",
                y="Winner")
        st.plotly_chart(fig)
    except:
        st.write("ERROR")
