"""

This script is used to performe an exploratory data analysis of both the dataset:
- teams_stats_2003-2004_2023-2024
- players_stats_2003-2004_2023-2024

"""

import pandas as pd
import matplotlib.pyplot as plt
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
output_path = os.path.join(script_path, "../output/")
figures_path = os.path.join(output_path, "figures")

# methods
def read_dataset(dataset):

    """
    
    This methods is used to read the csv file

    - dataset (str): which dataset to use 

    return:
    - df: pandas dataframe

    """
    print(path)
    if(dataset == "teams"):
        pass
    elif(dataset == "players"):
        pass

# main 
if(__name__ == "__main__"):
    pass