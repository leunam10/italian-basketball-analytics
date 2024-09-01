
"""

This script is used to create a streamlit dashboard to show the hystorical and present
data of the teams and players from 2003-2004 to 2023-2024 seasons.

"""



import pandas as pd
import streamlit as st
from streamlit_theme import st_theme
import plotly.express as px
import plotly.graph_objects as go
import copy
import os


path = os.path.dirname(os.path.abspath("lba_dashboard.py"))
data_path = os.path.join(path, "data")

@st.cache_data
def read_md_file():
    
    """
    
    This methods allows to read an markdown file containing the description of the
    statistics dataset.

    return
    - stats_help_md (str): markdown formatted string

    """
    
    with open("data/stats_help.md", "r") as file:
        stats_help_md = file.read()

    return stats_help_md

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
        

def df_selector(df, dataset, dataset_element, season_interval):

    """

    This method allows to select a dataframe with respect the specific player or team
    
    - df: pandas dataframe to select
    - dataset (str): which dataset will be selected
    - dataset_element (dict): the values of the dict are the name of the player or the team
    - season_interval (tuple): the interval of year to consider

    return
    - select_df: a pandas dataframe selection of the input dataframe

    """    

    # select by team/player. The Year is selected later
    if(dataset_element != "all" and len(dataset_element)>0):
        if(dataset == "LBA Teams"):
            select_df = df[df["Team"].isin(dataset_element)]
        elif(dataset == "LBA Players"):
            select_df = df[df["Player"].isin(dataset_element)]
    elif(dataset_element == "all" or len(dataset_element)==0):
        select_df = copy.copy(df)
            
    
    # select by year
    select_df = select_df.loc[(select_df["Year"]>=season_interval[0]) & (select_df["Year"]<=season_interval[1])]

    return select_df


def generic_metric_plot(df, dataset, metric):

    """

    This method is used to create a bar chart for the selected metric of the team or of the player

    - df (pandas dataframe)
    - dataset (str): which dataset will be selected
    - metric (str): which metric to show 

    """


    if(dataset == "LBA Teams"):    
        
        # create the dataframe of the count with respect the metric
        df_metric_count = df.groupby("Team")[metric].sum().reset_index()

        # select only the teams with metric larger than 1
        df_metric_count_filter = df_metric_count[df_metric_count[metric] > 0]

        # order the df with respect to the metric
        sort_df = df_metric_count_filter.sort_values(by=metric, ascending=False)

        # create the plotly figure
        fig = px.bar(sort_df, x="Team", y=metric)
        
        # update the y-label title
        if(metric == "Playoff"):
            fig.update_layout(yaxis_title="Number of playoff parteciations")
        elif(metric == "Finalist"):
            fig.update_layout(yaxis_title="Number of finals played")
        elif(metric == "Winner"):
            fig.update_layout(yaxis_title="Number of won seasons")

    elif(dataset == "LBA Players"):

        # create the dataframe of the count with respect the metric
        df_metric_count = df.groupby("Player")[metric].sum().reset_index()

        # select only the teams with metric larger than 1
        df_metric_count_filter = df_metric_count[df_metric_count[metric] > 0]

        # order the df with respect to the metric
        sort_df = df_metric_count_filter.sort_values(by=metric, ascending=False)

        # create the plotly figure
        fig = px.bar(sort_df, x="Player", y=metric)

        # update the y-label title
        if(metric == "MVP"):
            fig.update_layout(yaxis_title="Number of time MVP")


    # add the values of the bar on top of the bar
    fig.update_traces(text=sort_df[metric],  textposition='outside')

    # Adjust the y-axis limits (ylim)
    fig.update_yaxes(range=[0, sort_df[metric].max()+2], tickmode='linear', dtick=2)  # Adjust this range as needed 


    st.plotly_chart(fig)


def stats_plot(df, dataset, stat, year_for_bar_chart, sub_selection):

    """

    This method is used to plot a specific statistics for the team/player 
    dataframe. 

    - df (pandas dataframe)
    - dataset (str): which dataset will be selected
    - stat (str): which statistic to show
    - year_for_bar_chart (int): season to show in the bar chart
    - sub_selection (str): if to show a specific sub-samples of teams/players

    """

    if(dataset == "LBA Teams"):
        trace_name_list = list(df["Team"].unique())
        col_name = "Team"
    elif(dataset == "LBA Players"):
        trace_name_list = list(df["Player"].unique())
        col_name = "Player"

    df = df.sort_values(by="Year")

    fig = go.Figure()

    traces_list = []
    for trace_i in trace_name_list:

        select_df = df.loc[df[col_name]==trace_i]
        season_df = df.loc[(df["Year"]==year_for_bar_chart) & (df[col_name]==trace_i)]

        if(sub_selection == None): 
            trace = go.Bar(y=season_df[col_name], x=season_df[stat], name=trace_i, 
                           text=season_df[stat], textposition="outside", orientation = "h")
            traces_list.append(trace)
        else:
            if(sub_selection != None):
                try:
                    if(season_df[sub_selection].iloc[0] == 1):                         
                        trace = go.Bar(y=season_df[col_name], x=season_df[stat], name=trace_i, text=season_df[stat], textposition="inside", 
                                   marker_color=["red"], orientation="h")
                    else:
                        trace = go.Bar(y=season_df[col_name], x=season_df[stat], name=trace_i, text=season_df[stat], textposition="inside", 
                                       marker_color=["gray"], orientation="h")
                    traces_list.append(trace)
                except:
                    pass
        

    fig = go.Figure(data=traces_list)
    fig.update_layout(yaxis_title=stat)
        
    st.plotly_chart(fig)

def stats_aggregation(df, player, year):

    """
        
    This methods allows to aggregate the statistics of a player to have an overall indication
    of his performances.

    - df: pandas dataframe of all the players
    - player (str): name of the player
    - year (str): year over wich compute the performance
    
    return
    - performance_dict (dict): dict of the performance for the player

    """ 

    # dictionary definition
    performance_dict = {"Shooting" : 0,
                        "Scoring" : 0,
                        "Offensive Aggressiveness" : 0,
                        "Difensive Aggressiveness" : 0,
                        "Playmaking" : 0}
    
    # select the dataframe by year
    df_for_year = df.loc[df["Year"] == int(year)]       
    # select the player
    player_df = df.loc[(df["Player"] == player) & (df["Year"] == int(year))]

    # Shooting
    ## this is computed by consider the points made by the player normalize to maxiumum points made by a player during the season
        
    # compute the maximum value for the points
    points_max = df_for_year["PPG"].max()

    # compute the shooting performance
    shooting = int((player_df["PPG"].iloc[0]/points_max) * 100)

    # update the dictionary
    performance_dict["Shooting"] = shooting

    # Scoring
    ## this is computed by considering the normalized average value of the fg%(2pt% and 3pt%) and ft%
        
    # compute the scoring performance
    scoring = int((player_df["FG%"].iloc[0] + player_df["FT%"].iloc[0])/2*100)
        
    # update the dictionary
    performance_dict["Scoring"] = scoring

    # Offensive Aggressiveness
    ## this is computed by considering the offensive rebounds with respect the maximum value within the same year
    # compute the maximum value for the points
    orb_max = df_for_year["ORB"].max()

    # compute the shooting performance
    aggr_off = int((player_df["ORB"].iloc[0]/orb_max) * 100)

    # update the dictionary
    performance_dict["Offensive Aggressiveness"] = aggr_off

    # Difensive Aggressiveness
    ## this is computed as the average of the steal, blocks, fouls and difensive rebounds
    aggr_dif = int((player_df["SPG"].iloc[0]/df_for_year["SPG"].max() + 
                    player_df["BPG"].iloc[0]/df_for_year["BPG"].max() + 
                    player_df["PF"].iloc[0]/df_for_year["PF"].max() + 
                    player_df["DRB"].iloc[0]/df_for_year["DRB"].max())/4*100)

    # update dictionary
    performance_dict["Difensive Aggressiveness"] = aggr_dif

    # Playmaking
    ## this is computed as the average of assists and turnovers
    playmaking = int((player_df["APG"].iloc[0]/df_for_year["APG"].max() + 
                      player_df["TOV"].iloc[0]/df_for_year["TOV"].max())/2*100)

    # update dictionary
    performance_dict["Playmaking"] = playmaking

    return performance_dict


def player_performance_radar_chart(df, player, year, app_theme):

    """
    """

    if(app_theme == "light"):
        bgcolor = "white"
    else:
        bgcolor = "black"

    performance_dict = stats_aggregation(df, player, year)
    theta = list(performance_dict.keys())
    theta.append(theta[0])
    values = list(performance_dict.values())
    values.append(values[0])

    fig = go.Figure(
        data=go.Scatterpolargl(
            r=values,
            theta=theta,
            fill='toself',
            marker=dict(size=10, color="mediumseagreen")
        )
    )

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor = bgcolor))
    st.plotly_chart(fig)



if(__name__ == "__main__"):

    # get info about the streamlit theme selected
    app_theme = st_theme()["base"]
    
    # read csv files
    teams_df = read_data("teams")
    players_df = read_data("players")
    stats_help_md = read_md_file()

    # default values
    data_choice = "LBA Teams"
    if(data_choice == "LBA Teams"):
        start_year = teams_df["Year"].min()
        end_year = teams_df["Year"].max()
    elif(data_choice == "LBA Players"):
        start_year = players_df["Year"].min()
        end_year = players_df["Year"].max()

    ###########
    # Sidebar #
    ###########
    
    ## dataset selection
    st.sidebar.header("Dataset Selection")
    data_choice = st.sidebar.radio("Select the data you want to analyze",
                                  ["LBA Teams", "LBA Players"])
        
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
    #st.sidebar.header("Statistics Selection")
    #if(data_choice == "LBA Teams"):
    #    teams_cols = teams_df.columns
    #    cols_to_remove = ["Team", "Year", "Playoff", "Finalist", "Winner"]
    #    teams_stats = list(filter(lambda x: x not in cols_to_remove, teams_cols))
    #    stat = st.sidebar.selectbox("Select the statistics you want to show",
    #                                       (teams_stats), index=None,
    #                                       placeholder="Select the statistic...", help=stats_help_md)
    #elif(data_choice == "LBA Players"):
    #    players_cols = players_df.columns
    #    cols_to_remove = ["Player", "Year", "Team", "MVP"]
    #    players_stats = list(filter(lambda x: x not in cols_to_remove, players_cols))
    #    stat = st.sidebar.selectbox("Select the statistics you want to show",
    #                                       (players_stats), index=None,
    #                                       placeholder="Select the statistic...", help=stats_help_md)


    ## select the dataframe
    if(select_all_toggle): 
        if(data_choice == "LBA Teams"):
            if(dataset_element == "all"):
                select_df = df_selector(teams_df, data_choice, "all", season_interval)
        elif(data_choice == "LBA Players"):
            if(dataset_element == "all"):
                select_df = df_selector(players_df, data_choice, "all", season_interval)
    else:
        if(data_choice == "LBA Teams"):
            if(len(dataset_element)>0 or season_interval):
                select_df = df_selector(teams_df, data_choice, dataset_element, season_interval)
        elif(data_choice == "LBA Players"):
            if(len(dataset_element or season_interval)>0):
                select_df = df_selector(players_df, data_choice, dataset_element, season_interval)

    select_df = select_df.sort_values(by="Year")


    ##################
    # main dashboard #
    ##################

    ## setup the title
    st.title("🏀 Italian Basketball A League Analytics")
    
    ## show the dataframe as a table
    if(data_choice == "LBA Teams"):
        st.header("LBA Teams Dataframe")
        try:
            st.dataframe(select_df)
        except:
            st.dataframe(teams_df)
    elif(data_choice == "LBA Players"):
        st.header("LBA Players Dataframe")
        try:
            st.dataframe(select_df)
        except:
            st.dataframe(players_df)

    ## Metrics
    if(data_choice == "LBA Teams"):
        st.header("Team Metrics")
        col1, col2, col3 = st.columns(3) 
        
        try:
            playoff_partecipations = select_df["Playoff"].sum()
            final_partecipations = select_df["Finalist"].sum()
            winned_season = select_df["Winner"].sum() 
            col1.metric("Number of playoff partecipation", playoff_partecipations)
            col2.metric("Number of final partecipation", final_partecipations)
            col3.metric("Number of seasons won", winned_season)
        except:
            col1.metric("Number of playoff partecipation", None)
            col2.metric("Number of final partecipation", None)
            col3.metric("Number of seasons won", None)

        metric = st.radio("Select the team metric", 
                          ["Playoff", "Finalist", "Winner"],
                          captions=["Number of playoff partecipations", "Number of finals played", "Number of won seasons"])

        try:
            generic_metric_plot(select_df, data_choice, metric)
        except:
            st.write("**Select at least one team to show the metric chart**")

    elif(data_choice == "LBA Players"):
        st.header("Player Metrics")
        metric = "MVP"
        try:
            mvp_winner = select_df["MVP"].sum()
            st.metric("Number of times MVP", mvp_winner)   
        except:
            st.metric("Number of times MVP", None)    

        generic_metric_plot(select_df, data_choice, metric)
        
    
    ## statistic plots
    st.header("Statistics Chart")
    
    if(data_choice == "LBA Teams"):
        teams_cols = teams_df.columns
        cols_to_remove = ["Team", "Year", "Playoff", "Finalist", "Winner"]
        teams_stats = list(filter(lambda x: x not in cols_to_remove, teams_cols))
        stat = st.selectbox("Select the statistics you want to show",
                                           (teams_stats), index=None,
                                           placeholder="Select the statistic...", help=stats_help_md)
    elif(data_choice == "LBA Players"):
        players_cols = players_df.columns
        cols_to_remove = ["Player", "Year", "Team", "MVP"]
        players_stats = list(filter(lambda x: x not in cols_to_remove, players_cols))
        stat = st.selectbox("Select the statistics you want to show",
                                           (players_stats), index=None,
                                           placeholder="Select the statistic...", help=stats_help_md)

    year_for_bar_chart = st.selectbox("Select the season you want to use for the bar chart",
                                           (select_df["Year"].unique()), index=None,
                                           placeholder="Select the season...", key="bar")


    if(data_choice == "LBA Teams"):
        sub_selection = st.selectbox("Chart sub-selection", 
                                      ("Playoff", "Finalist", "Winner"), index=None)

    elif(data_choice == "LBA Players"):
        sub_selection = st.selectbox("Chart sub-selection", 
                                      ("MVP"), index=None)

    if(stat != None):
        stats_plot(select_df, data_choice, stat, year_for_bar_chart, sub_selection)
    else:
        st.write("**Select at least one statistic to show the chart**")
    

    # Player Performance analyzer
    if(data_choice == "LBA Players"):
        st.header("Player Performance Analyzer")

        player = st.selectbox("Select the player you want to analyze", 
                               list(select_df["Player"].unique()), placeholder="Select the player...",
                               index=None)

        player_df = players_df.loc[players_df["Player"] == player]
        year_for_radar_chart = st.selectbox("Select the season you want to use for the bar chart",
                                           (player_df["Year"].unique()), index=None,
                                           placeholder="Select the season...", key="radar")
        try:
            player_performance_radar_chart(select_df, player, year_for_radar_chart, app_theme)       
        except:
            st.write("**Select a player to show the chart**")          