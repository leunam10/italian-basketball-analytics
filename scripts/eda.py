"""

This script is used to performe an exploratory data analysis of both the dataset:
- teams_stats_2003-2004_2023-2024
- players_stats_2003-2004_2023-2024

"""

import numpy as np
import pandas as pd
import itertools
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import os

# Argument Definition 
parser = argparse.ArgumentParser(
    prog = "Exploratory Data Analysis (EDA) script",
    description = "This script is used to performe an exploratory data analysis of the teams and players datasets"
)

parser.add_argument("--dataset", choices=["teams", "players"], help = "which dataset to use for the analysis")
args = parser.parse_args()

dataset = args.dataset

# path definition
script_path = os.path.dirname(os.path.abspath("eda.py"))
data_path = os.path.join(script_path, "../data/")
output_path = os.path.join(script_path, "../output/")
figures_path = os.path.join(output_path, "figures/")

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
    if "Team" in df.columns and "Player" not in df.columns:
        df = df.drop(columns=['Team'])
        if label_column == "Playoff":
            df = df.drop(columns=["Finalist", "Winner"])
        elif label_column == "Finalist":
            df = df.drop(columns=["Playoff", "Winner"])
        elif label_column == "Winner":
            df = df.drop(columns=["Playoff", "Finalist"])
    elif "Player" in df.columns:
        df = df.drop(columns=["Player", "Team"])        

    # drop GP
    df = df.drop(columns=["GP"])

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
        plt.savefig(os.path.join(figures_path,savename), bbox_inches='tight', dpi=300)
        plt.close()

def plot_feature_relationship(df, feature_x, feature_y, savename=None):
    """
    Plots one feature against another from a pandas DataFrame with colors representing different years.
    Optionally saves the figure.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature_x (str): The name of the feature to be plotted on the x-axis.
        feature_y (str): The name of the feature to be plotted on the y-axis.
        savename (str, optional): The file path to save the plot. If None, the plot is not saved.

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
    
    unique_years = df['Year'].unique()
    palette = sns.color_palette("viridis", len(unique_years))

    # Create the plot
    plt.figure(figsize=(12, 8))
    scatter_plot = sns.scatterplot(
        x=feature_x,
        y=feature_y,
        data=df,
        hue="Year",  # Color points by year
        palette=palette,  # Color palette
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
        plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        plt.close()

def plot_column_distribution(df, column_name, plot_type='hist', bins=20, kde=True, savename=None):
    """
    Plots the distribution of a specified column in a pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        column_name (str): The name of the column to plot the distribution of.
        plot_type (str): The type of plot to display ('hist' for histogram, 'kde' for density plot).
        bins (int): The number of bins for the histogram (if plot_type is 'hist').
        kde (bool): Whether to overlay a Kernel Density Estimate (KDE) on the histogram (if plot_type is 'hist').
        save_path (str, optional): The file path to save the plot. If None, the plot is not saved.
        
    Returns:
        None: The function displays the plot and optionally saves it.
    """
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' is not in the DataFrame.")
    
    # Set plot style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))

    if plot_type == 'hist':
        # Histogram with optional KDE overlay
        sns.histplot(df[column_name], bins=bins, kde=kde, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {column_name}', fontsize=16)
    elif plot_type == 'kde':
        # Kernel Density Estimate (KDE) plot
        sns.kdeplot(df[column_name], shade=True, color='skyblue')
        plt.title(f'Density Plot of {column_name}', fontsize=16)
    else:
        raise ValueError("Invalid plot_type. Choose 'hist' for histogram or 'kde' for density plot.")
    
    # Add labels
    plt.xlabel(column_name, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Save the plot if a save path is provided
    if savename:
        plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        plt.close()

def plot_distribution_by_year(df, column_name, bins=20, savename=None):
    """
    Plots the distribution of a specified column in a pandas DataFrame grouped by 'Year'.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        column_name (str): The name of the column to plot the distribution of.
        bins (int): The number of bins for the histogram.
        savename (str, optional): The file path to save the plot. If None, the plot is not saved.
        
    Returns:
        None: The function displays the plot and optionally saves it.
    """
    # Check if the specified column and 'Year' exist in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' is not in the DataFrame.")
    
    if 'Year' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'Year' column.")

    # Ensure 'Year' is treated as a categorical variable for FacetGrid
    df = df.sort_values('Year')
    df['Year'] = df['Year'].astype(str)

    # Set up the FacetGrid with a distinct color for each year
    unique_years = sorted(df['Year'].unique())
    palette = sns.color_palette("husl", len(unique_years))

    # Set up the FacetGrid
    g = sns.FacetGrid(df, col="Year", col_wrap=4, height=4, sharex=False, sharey=False, palette=palette)
    
    # Map a histogram onto each facet
    g.map(sns.histplot, column_name, bins=bins, kde=True, edgecolor='black')
    
    # Add titles to each facet
    g.set_titles(col_template="{col_name} Year")
    
    # Adjust the layout
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Distribution of {column_name} Over Different Years', fontsize=16)

    # Save the plot if a save path is provided
    if savename:
        plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        plt.close()

def plot_feature_by_year_and_team(df, feature, teams=None, players=None, savename=None):
    """
    Plots a specific feature for different years and different teams.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature (str): The name of the feature/column to plot.
        teams (list): Optional. A list of team names to plot. If None, all teams are plotted.
        players (list): Optional. A list of player names to plot. If None, all teams are plotted.
        savename (str): Optional. The path to save the plot image. If None, the plot is not saved.

    Returns:
        None
    """
    # Ensure the required columns are in the DataFrame
    if 'Year' not in df.columns or 'Team' not in df.columns or feature not in df.columns:
        raise ValueError("The DataFrame must contain 'Year', 'Team', and the specified feature columns.")

    # Filter the DataFrame for the specified teams (if provided)
    if teams:
        df = df[df['Team'].isin(teams)]
        players = None
    if players:
        df = df[df["Player"].isin(players)]
        teams = None

    # Ensure the Year column is treated as numeric for proper plotting
    df.loc[:, 'Year'] = pd.to_numeric(df['Year'])

    # Plot the feature by year for different teams
    plt.figure(figsize=(12, 6))
    if teams:
        sns.lineplot(data=df, x='Year', y=feature, hue='Team', marker='o', palette='tab10')
    if players:
        sns.lineplot(data=df, x='Year', y=feature, hue='Player', marker='o', palette='tab10')

    # Set plot labels and title
    plt.xlabel('Year')
    plt.ylabel(feature)
    plt.title(f'{feature} by Year for Different Teams')
    
    # Show legend and format the x-axis to display as integer
    if teams:
        plt.legend(title='Teams')
    if players:
        plt.legend(title='Players')

    plt.xticks(sorted(df['Year'].unique()))  # Ensure the x-axis is sorted and shows all years
    
    # Show or save the plot
    if savename:
        plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_teams_players_distribution(df, column='Team', savename=None):
    """
    Plots the distribution of teams or players in the dataset.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        column (str): The column to plot the distribution for ('Team' or 'Player').
        savename (str): Optional. The path to save the plot image. If None, the plot is not saved.

    Returns:
        None
    """
    # Ensure the required column is in the DataFrame
    if column not in df.columns:
        raise ValueError(f"The DataFrame must contain a '{column}' column.")

    # Plot the distribution of the specified column (Team or Player)
    plt.figure(figsize=(12, 6))
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']
    sns.barplot(x=column, y='Count', data=value_counts, hue=column, palette='tab20', dodge=False, legend=False)
    # Annotate bars with the count values
    for index, row in value_counts.iterrows():
        plt.text(index, row['Count'] + 0.5, f"{row['Count']}", color='black', ha="center")

    # Set plot labels and title
    plt.xticks(rotation=90)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}s in the Dataset')
    
    # Show or save the plot
    if savename:
        plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_value_counts(df, column_name, savename=None):
    """
    Counts the occurrences of each unique value in a specified column of a pandas DataFrame
    and creates a bar plot of these counts.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        column_name (str): The name of the column to count the occurrences of.
        savename (str, optional): The file path to save the plot. If None, the plot is not saved.
        
    Returns:
        None: The function displays the plot and optionally saves it.
    """
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' is not in the DataFrame.")
    
    # Count occurrences of each unique value in the specified column
    value_counts = df[column_name].value_counts().reset_index()
    value_counts.columns = [column_name, 'Count']  # Rename columns for clarity

    # Set plot style
    sns.set(style="whitegrid")

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    barplot=sns.barplot(
        x=column_name, 
        y='Count', 
        data=value_counts, 
        hue=column_name,  # Assign the `x` variable to `hue`
        edgecolor="k",
        palette="husl", 
        dodge=False, 
        legend=False  # Disable the legend
    )

    # Add the height of each bar on top
    for index, row in value_counts.iterrows():
        barplot.text(
            x=index, 
            y=row['Count'], 
            s=f'{row["Count"]}', 
            ha='center', 
            va='bottom', 
            fontsize=12
        )

    # Add titles and labels
    plt.title(f'Count of Each Unique Value in {column_name}', fontsize=16)
    plt.xlabel(column_name, fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # Rotate x-axis labels if they are long
    plt.xticks(rotation=45)

    # Save the plot if a savename is provided
    if savename:
        plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        plt.close()

def perform_pca(X, y, n_components=2, savename=None):
    """
    Performs Principal Component Analysis (PCA) on the given feature array X
    and plots the first two principal components, colored by discrete labels in y.
    
    Parameters:
        X (np.ndarray): The feature array of shape (n_samples, n_features).
        y (np.ndarray): The label array of shape (n_samples,), with discrete values.
        n_components (int): Number of principal components to keep (default is 2).
        savename (str, optional): The file path to save the plot. If None, the plot is not saved.
    
    Returns:
        X_pca (np.ndarray): Transformed feature array after PCA.
    """
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Performing PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Creating a discrete color palette
    unique_labels = np.unique(y)
    palette = sns.color_palette("husl", len(unique_labels))

    # Plotting the first two principal components
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(unique_labels):
        plt.scatter(
            X_pca[y == label, 0], 
            X_pca[y == label, 1], 
            color=palette[i], 
            label=f'Class {label}', 
            alpha=0.8, 
            edgecolor='k'
        )

    # Adding labels, title, and legend
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.title('PCA of the features', fontsize=16)
    plt.legend(title='Class', loc='best', fontsize=12)
    
    # Save the plot if a savename is provided
    if savename:
        plt.savefig(savename, bbox_inches='tight', dpi=300)
        plt.close()

    return X_pca

def compute_and_plot_correlations(df, label_column, method='pearson', plot=True, savename=None):
    """
    Computes the correlation matrix of each column with every other column in a DataFrame.
    Moves the label column to the last row and last column in the heatmap.
    Optionally plots a heatmap of the correlation matrix.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        label_column (str): The name of the label column to be moved to the last row/column.
        method (str): The method of correlation ('pearson', 'spearman', 'kendall'). Default is 'pearson'.
        plot (bool): Whether to plot a heatmap of the correlation matrix. Default is True.
        savename (str, optional): The file path to save the heatmap plot. If None, the plot is not saved.
    
    Returns:
        corr_matrix (pd.DataFrame): The correlation matrix with the label column at the last row/column.
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

    # drop GP
    df = df.drop(columns=["GP"])

    # Ensure the label column is moved to the end
    if label_column in df.columns:
        # Reorder the columns to move the label column to the end
        df = df[[col for col in df.columns if col != label_column] + [label_column]]
    else:
        raise ValueError(f"Label column '{label_column}' not found in the DataFrame.")

    # Compute the correlation matrix
    corr_matrix = df.corr(method=method)

    # Reorder the rows to move the label column to the end
    corr_matrix = corr_matrix.loc[corr_matrix.index != label_column]._append(corr_matrix.loc[label_column])

    # Plot the correlation matrix as a heatmap if requested
    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', cbar=True, 
                    annot_kws={"size": 8}, linewidths=.5)

        # Adjust the text and tick parameters for readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=16)

        # Save the plot if a savename is provided
        if savename:
            plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
            print(f"Heatmap saved to {savename}")


    return corr_matrix

def plot_correlations_as_bar(df, label_column, method='pearson', savename=None):
    """
    Computes the correlation of a specific column (label) with all other columns in a DataFrame
    and visualizes it using a bar plot.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        label_column (str): The name of the label column to compute correlations with.
        method (str): The method of correlation ('pearson', 'spearman', 'kendall'). Default is 'pearson'.
        savename (str, optional): The file path to save the bar plot. If None, the plot is not saved.
    
    Returns:
        correlations (pd.Series): A Series of correlations of the label column with other columns.
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

    # drop GP
    df = df.drop(columns=["GP"])

    # Ensure the label column exists in the DataFrame
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the DataFrame.")

    # Compute the correlation matrix
    corr_matrix = df.corr(method=method)

    # Extract correlations of the specified label column with all other columns
    correlations = corr_matrix[label_column].drop(label_column)  # Remove the label's self-correlation

    # Sort correlations by value (both positive and negative)
    correlations_sorted = correlations.sort_values(ascending=False).reset_index()
    correlations_sorted.columns = ['Feature', 'Correlation']

    # Plot the correlations as a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Feature', 
        y='Correlation', 
        hue='Feature',  # Set hue to 'Feature' to apply palette without legend
        data=correlations_sorted, 
        palette="coolwarm", 
        dodge=False,  # Disable dodging since hue is just for color application
        legend=False  # Disable legend since each bar is already labeled by x-axis
    )
    
    # Annotate the bars with their correlation values
    for index, value in enumerate(correlations_sorted['Correlation'].values):
        plt.text(index, value, f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
    
    # Customize plot appearance
    plt.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0 for reference
    plt.title(f'Correlation of Features with {label_column} ({method.capitalize()})', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Correlation', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Save the plot if a savename is provided
    if savename:
        plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        print(f"Bar plot saved to {savename}")
    
    # Show the plot
    plt.show()

    return correlations_sorted

def kruskal_wallis_test(df, label_column, significance_level=0.05):
    """
    Performs the Kruskal-Wallis test to estimate the significance of all features in the DataFrame
    for predicting the label, and returns the number of significant features.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        label_column (str): The name of the label column.
        significance_level (float): The threshold for significance (default is 0.05).
    
    Returns:
        (pd.DataFrame, int): A tuple containing a DataFrame with features and their corresponding
                             Kruskal-Wallis test statistics and p-values, and the number of significant features.
    """
    # Ensure the label column exists in the DataFrame
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the DataFrame.")
    
    # Automatically select all feature columns except the label column
    features = [col for col in df.columns if col != label_column]
    
    # Initialize lists to store results
    results = {'Feature': [], 'H-statistic': [], 'p-value': []}
    
    # Unique values of the label
    label_values = df[label_column].unique()

    # Perform Kruskal-Wallis test for each feature
    for feature in features:
        # Prepare the data for Kruskal-Wallis test
        groups = [df[df[label_column] == label_value][feature].dropna() for label_value in label_values]
        
        # Perform Kruskal-Wallis test
        h_statistic, p_value = kruskal(*groups)
        
        # Store the results
        results['Feature'].append(feature)
        results['H-statistic'].append(h_statistic)
        results['p-value'].append(p_value)
    
    # Convert the results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)
    results_df.sort_values('p-value', ascending=True, inplace=True)
    
    # Count the number of significant features
    significant_features_count = (results_df['p-value'] < significance_level).sum()

    print(f"Number of significant features (p < {significance_level}): {significant_features_count}")

    return results_df, significant_features_count


# main 
if(__name__ == "__main__"):

    df = read_csv_as_dataframe(dataset)
    print(df.head())

    # select features for plot functions
    if dataset == "teams":
        features_list = [col for col in df.columns if col!="Team" 
                        and col!="Playoff" and col!="Finalist" and col!="Winner" 
                        and col!="GP"] 
    elif dataset == "players":
        features_list = [col for col in df.columns if col!="Player" and col!="MVP" and col!="Team"]

    # feature by year
    for feature in features_list:
        savename = f"{dataset}_{feature}_by_year.png"
        plot_column_by_year(df, feature, savename=savename)

    # Generate all combinations of features
    feature_pairs = list(itertools.combinations(features_list, 2))
    # features relationship
    for feature_1, feature_2 in feature_pairs:
        savename = f"{dataset}_{feature_1}_vs_{feature_2}.png"
        plot_feature_relationship(df, feature_1, feature_2, savename=savename)

    # features distribution
    for feature in features_list:
        savename = f"{dataset}_{feature}_distribution.png"
        plot_column_distribution(df, feature, plot_type='hist', bins=20, kde=True, savename=savename)

    # features distribution by year
    for feature in features_list:
        savename = f"{dataset}_{feature}_distribution_by_year.png"
        plot_distribution_by_year(df, feature, bins=20, savename=savename)

    # features disribution by year and team
    for feature in features_list:
        savename = f"{dataset}_{feature}_distribution_by_year_and_team.png"
        if dataset == "teams":
            plot_feature_by_year_and_team(df, "APG", teams=df["Teams"].unique(), players=None, savename=savename)
        if dataset == "players":
            plot_feature_by_year_and_team(df, "APG", teams=None, players=df["Player"].unique(), savename=savename)

    # teams and players distribution
    if dataset == "teams":
        savename = f"{dataset}_distribution.png"
        plot_teams_players_distribution(df, column='Team', savename=savename)
    if dataset == "players":
        savename = f"{dataset}_distribution.png"
        plot_teams_players_distribution(df, column='Player', savename=savename)

    # label counts
    if dataset == "teams":
        plot_value_counts(df, "Playoff", savename="teams_playoff_label_counts.png")
        plot_value_counts(df, "Finalist", savename="teams_finalist_label_counts.png")
        plot_value_counts(df, "Winner", savename="teams_winner_label_counts.png")
    if dataset == "players":
        plot_value_counts(df, "MVP", savename="players_mvp_label_counts.png")

    # pca plot
    if dataset == "teams":
        X, y = dataframe_to_numpy(df, "Playoff")
        _ = perform_pca(X, y, n_components=2, savename=f"{dataset}_pca_2d_playoff.png")
        X, y = dataframe_to_numpy(df, "Finalist")
        _ = perform_pca(X, y, n_components=2, savename=f"{dataset}_pca_2d_finalist.png")
        X, y = dataframe_to_numpy(df, "Winner")
        _ = perform_pca(X, y, n_components=2, savename=f"{dataset}_pca_2d_winner.png")
    if dataset == "players":
        X, y = dataframe_to_numpy(df, "MVP")
        _ = perform_pca(X, y, n_components=2, savename=f"{dataset}_pca_2d_mvp.png")

    # pearson correlation
    if dataset == "teams":
        compute_and_plot_correlations(df, "Playoff", method='pearson', plot=True, savename="teams_correlation_with_playoff_label.png")
        compute_and_plot_correlations(df, "Finalist", method='pearson', plot=True, savename="teams_correlation_with_finalist_label.png")
        compute_and_plot_correlations(df, "Winner", method='pearson', plot=True, savename="teams_correlation_with_winner_label.png")

        plot_correlations_as_bar(df, "Playoff", method='pearson', savename="teams_bar_correlation_with_playoff_label.png")
        plot_correlations_as_bar(df, "Finalist", method='pearson', savename="teams_bar_correlation_with_finalist_label.png")
        plot_correlations_as_bar(df, "Winner", method='pearson', savename="teams_bar_correlation_with_winner_label.png")

    if dataset == "players":
        compute_and_plot_correlations(df, "MVP", method='pearson', plot=True, savename="players_correlation_with_mvp_label.png")
        plot_correlations_as_bar(df, "MVP", method='pearson', savename="players_bar_correlation_with_mvp_label.png")

    #a, b = kruskal_wallis_test(df, "Playoff")