"""

This script is used to performe an exploratory data analysis of both the dataset:
- teams_stats_2003-2004_2023-2024
- players_stats_2003-2004_2023-2024

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Argument Definition 
parser = argparse.ArgumentParser(
    prog = "Exploratory Data Analysis (EDA) script",
    description = "This script is used to performe an exploratory data analysis of the teams and players datasets"
)

parser.add_argument("--dataset", choices=["teams", "players"], help = "which dataset to use for the analysis")
parser.add_argument("--savefigure", choices=["yes", "no"], default="no", help = "if to save of not the figures")

args = parser.parse_args()

dataset = args.dataset
savefigure = True if args.savefigure == "yes" else False

# path definition
script_path = os.path.dirname(os.path.abspath("eda.py"))
data_path = os.path.join(script_path, "../data/")
output_path = os.path.join(script_path, "../output/")
figures_path = os.path.join(output_path, "figures")

# methods
def read_csv_as_dataframe(dataset):
    """
    Reads a CSV file as a pandas DataFrame based on the 'dataset' input.

    Parameters:
        dataset (str): Specifies the dataset type, either 'teams' or 'players'.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the CSV data.
        
    Raises:
        ValueError: If the 'dataset' input is not 'teams' or 'players'.
    """
    
    # Determine the file path based on the input dataset
    if dataset == "teams":
        file_path = os.path.join(data_path, "teams_stats_2003-2004_2023-2024.csv")  # Replace with your actual file path
    elif dataset == "players":
        file_path = os.path.join(data_path, "players_stats_2003-2004_2023-2024.csv")  # Replace with your actual file path
    else:
        raise ValueError("Invalid dataset type. Choose 'teams' or 'players'.")
    
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file '{file_path}' could not be parsed.")

    return df

def dataframe_to_numpy(df, label_column):
    """
    Converts a pandas DataFrame into a NumPy array, separating features and label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): The name of the column to be used as the label (y).

    Returns:
        X (np.ndarray): A NumPy array containing the features.
        y (np.ndarray): A NumPy array containing the label.
    """
    
    # Drop 'Team' or 'Player' column if present
    if "Team" in df.columns:
        df = df.drop(columns=['Team'])
        if label_column == "Playoff":
            df = df.drop(columns=["Finalist", "Winner"])
        elif label_column == "Finalist":
            df = df.drop(columns=["Playoff", "Winner"])
        elif label_column == "Winner":
            df = df.drop(columns=["Playoff", "Finalist"])
    elif "Player" in df.columns:
        df = df.drop(columns=["Player", "Team"])        

    # Check if the label_column is in the DataFrame
    if label_column not in df.columns:
        raise ValueError(f"The column '{label_column}' is not in the DataFrame.")
    
    # Separate the label (y) from the features (X)
    y = df[label_column].values  # Convert the label column to a NumPy array
    X = df.drop(columns=[label_column]).values  # Drop the label column and convert the remaining to a NumPy array

    return X, y    

def plot_column_by_year(df, column, savename=None):
    """
    Plots a specified column as a function of "Year" from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column to plot against "Year".

    Returns:
        None: The function displays the plot.
    """
    # Check if the specified column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"The column '{column}' is not in the DataFrame.")
    
    # Check if the "Year" column exists in the DataFrame
    if "Year" not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Year' column.")

    # Set plot style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Year", y=column, data=df, s=100, color="blue")  
    sns.lineplot(x="Year", y=column, data=df, marker="o", color="red")

    # Ensure x-axis is treated as integer
    plt.xticks(ticks=range(df['Year'].min(), df['Year'].max() + 1), rotation=45)  

    # Add plot title and labels
    plt.title(f'{column} by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel(column, fontsize=14)

    # Show the plot
    plt.xticks(rotation=45)
    if savename:
        plt.savefig("prova.png", bbox_inches='tight', dpi=300)

def plot_feature_relationship(df, feature_x, feature_y, savename=None):
    """
    Plots one feature against another from a pandas DataFrame with colors representing different years.
    Optionally saves the figure.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature_x (str): The name of the feature to be plotted on the x-axis.
        feature_y (str): The name of the feature to be plotted on the y-axis.
        save_path (str, optional): The file path to save the plot. If None, the plot is not saved.

    Returns:
        None: The function displays the plot and optionally saves it.
    """
    # Check if the specified features and "Year" column exist in the DataFrame
    if feature_x not in df.columns or feature_y not in df.columns:
        raise ValueError(f"One or both of the features '{feature_x}' or '{feature_y}' are not in the DataFrame.")
    
    if "Year" not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Year' column.")
    

    # Set plot style
    sns.set(style="whitegrid")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    scatter_plot = sns.scatterplot(
        x=feature_x,
        y=feature_y,
        data=df,
        hue="Year",  # Color points by year
        palette="viridis",  # Color palette
        size="Year",  # Optional: change size of markers by year
        sizes=(50, 200),  # Size range for the markers
        marker='o'
    )

    # Add plot title and labels
    plt.title(f'{feature_y} vs {feature_x} by Year', fontsize=16)
    plt.xlabel(feature_x, fontsize=14)
    plt.ylabel(feature_y, fontsize=14)

    # Add legend for years
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot if a save path is provided
    if savename:
        plt.savefig("prova.png", bbox_inches='tight', dpi=300)
        
# main 
if(__name__ == "__main__"):

    df = read_csv_as_dataframe(dataset)
    print(df)
    
    plot_feature_relationship(df, "APG", "RPG", savename="prova")
