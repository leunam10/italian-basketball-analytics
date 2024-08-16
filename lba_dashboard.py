
"""

This script is used to create a streamlit dashboard to show the hystorical and present
data of the teams and players from 2003-2004 to 2023-2024 seasons.

"""



import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import copy
import os


path = os.path.dirname(os.path.abspath("lba_dashboard.py"))
data_path = os.path.join(path, "data")

@st.cache_data
def read_md_file(dataset):
    
    """
    
    This methods allows to read an markdown file containing the description of the
    statistics for team and player dataset.

    - dataset (str):  which dataset to load as pandas dataframe

    return
    - stats_help_md (str): markdown formatted string

    """
    if(dataset == "teams"):
        with open("data/players_stats_help.md", "r") as file:
            stats_help_md = file.read()
    elif(dataset == "players"):
        with open("data/teams_stats_help.md", "r") as file:
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


def stats_plot(df, dataset, stat, char_type, year_for_bar_chart):

    """

    This method is used to plot a specific statistics for the team/player 
    dataframe. 

    - df (pandas dataframe)
    - dataset (str): which dataset will be selected
    - stat (str): which statistic to show

    """

    if(dataset == "LBA Teams"):
        trace_name_list = list(df["Team"].unique())
        col_name = "Team"
    elif(dataset == "LBA Players"):
        trace_name_list = list(df["Player"].unique())
        col_name = "Player"

    df = df.sort_values(by="Year")

    if(char_type == "Line"):
        fig = go.Figure()

        traces_list = []
        for team in trace_name_list:
            df_team = df.loc[df[col_name]==team]

            trace = go.Scatter(x=df_team["Year"], y=df_team[stat], mode="lines+markers", name=team)
            traces_list.append(trace)

        fig = go.Figure(data=traces_list)

    elif(char_type == "Bar"):
        pass

    st.plotly_chart(fig)




if(__name__ == "__main__"):

    # read csv files
    teams_df = read_data("teams")
    players_df = read_data("players")
    
    teams_stats_help_md = read_md_file("teams")
    players_stats_help_md = read_md_file("players")

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
    st.sidebar.header("Statistics Selection")
    if(data_choice == "LBA Teams"):
        teams_cols = teams_df.columns
        cols_to_remove = ["Team", "Year", "Playoff", "Finalist", "Winner"]
        teams_stats = list(filter(lambda x: x not in cols_to_remove, teams_cols))
        stat = st.sidebar.selectbox("Select the statistics you want to show",
                                           (teams_stats), index=None,
                                           placeholder="Select the statistic...", help=teams_stats_help_md)
    elif(data_choice == "LBA Players"):
        players_cols = players_df.columns
        cols_to_remove = ["Player", "Year", "Team", "MVP"]
        players_stats = list(filter(lambda x: x not in cols_to_remove, players_cols))
        stat = st.sidebar.selectbox("Select the statistics you want to show",
                                           (players_stats), index=None,
                                           placeholder="Select the statistic...", help=players_stats_help_md)


    ## select the dataframe
    if(select_all_toggle): 
        if(data_choice == "LBA Teams"):
            if(dataset_element == "all"):
                select_df = df_selector(teams_df, data_choice, "all", season_interval, stat)
        elif(data_choice == "LBA Players"):
            if(dataset_element == "all"):
                select_df = df_selector(players_df, data_choice, "all", season_interval, stat)
    else:
        if(data_choice == "LBA Teams"):
            if(len(dataset_element)>0 or season_interval):
                select_df = df_selector(teams_df, data_choice, dataset_element, season_interval, stat)
        elif(data_choice == "LBA Players"):
            if(len(dataset_element or season_interval)>0):
                select_df = df_selector(players_df, data_choice, dataset_element, season_interval, stat)

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
    
    col1, col2 = st.columns(2)

    chart_type = col1.radio("Select the type of chart", ["Line", "Bar"])

    if(chart_type == "Bar"):
        year_selection = False
    else:
        year_selection = True


    year_for_bar_chart = col2.selectbox("Select the season you want to use for the bar chart",
                                           (select_df["Year"].unique()), index=None,
                                           placeholder="Select the season...", disabled=year_selection)


    try:
        stats_plot(select_df, data_choice, stat, chart_type, year_for_bar_chart)
    except:
        st.write("**Select at least one statistic to show the chart**")
    


  
