"""

This script is used to build  machine learning models capable of doing different prediction:

- team season winner
- team partecipating to the final of the playoff
- team partecipanting to the playoff

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import argparse
import joblib

import os


class MLPredictor:

    def __init__(self, data_path, output_path, figures_path, dataset_type):
        """
        Initialize the MLPredictor class with dataset path and type.

        Parameters:
            data_path (str): Path to the dataset file.
            output_path (str): Path to the output files.
            figures_path (str): Path to the figures files.
            dataset_type (str): The type of dataset ('teams' or 'players').
        """

        self.data_path = data_path
        self.output_path = output_path
        self.figures_path = figures_path
        self.dataset_type = dataset_type
        self.df = self.read_data()
    
    def read_data(self):
        """
        Reads the CSV file as a pandas DataFrame based on the dataset type.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self.dataset_type == 'teams':
            return pd.read_csv(self.data_path)
        elif self.dataset_type == 'players':
            return pd.read_csv(self.data_path)
        else:
            raise ValueError("Invalid dataset type. Choose 'teams' or 'players'.")

    def dataframe_to_numpy(self, df, label_column, features_to_drop=None):
        """
        Converts a pandas DataFrame into a NumPy array, separating features and label.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            label_column (str): The name of the column to be used as the label (y).
            features_to_drop (list): list of the features to drop before to create the array

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

        if features_to_drop:
            df = df.drop(columns=features_to_drop)

        # Check if the label_column is in the DataFrame
        if label_column not in df.columns:
            raise ValueError(f"The column '{label_column}' is not in the DataFrame.")

        # Separate the label (y) from the features (X)
        y = df[label_column].values  # Convert the label column to a NumPy array
        X = df.drop(columns=[label_column]).values  # Drop the label column and convert the remaining to a NumPy array

        return X, y    

    def standardize_features(self, X, method='standard'):
        """
        Standardizes the features using either StandardScaler or Min-Max scaling.

        Parameters:
            X (np.ndarray): Feature array to be standardized.
            method (str): The method to use for scaling ('standard' or 'min-max').

        Returns:
            X_scaled (np.ndarray): The standardized feature array.
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'min-max':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid method. Choose 'standard' or 'min-max'.")

        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def split_dataset(self, X, y, test_size=0.2, shuffle=True, random_state=42):
        """
        Splits the dataset into training and testing sets.

        Parameters:
            X (np.ndarray): Feature array.
            y (np.ndarray): Label array.
            test_size (float): Proportion of the dataset to include in the test split.
            shuffle (logic): if the shuffle the dataset before the split
            random_state (int): Random seed for reproducibility.

        Returns:
            X_train (np.ndarray): Training feature array.
            X_test (np.ndarray): Testing feature array.
            y_train (np.ndarray): Training label array.
            y_test (np.ndarray): Testing label array.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def perform_feature_selection(X, y, method='univariate', k=10, estimator=None, scale_features=True):
        """
        Perform feature selection using specified method and return the selected features.

        Parameters:
            X (np.ndarray or pd.DataFrame): Feature array.
            y (np.ndarray or pd.Series): Label array.
            method (str): Feature selection method ('univariate', 'rfe', 'importances').
            k (int): Number of top features to select.
            estimator: Estimator to use for RFE or feature importances. If None, defaults to LogisticRegression for RFE.
            scale_features (bool): If True, standardize the features before selection.

        Returns:
            np.ndarray: Selected feature indices.
            list: Names of selected features if X is a DataFrame.
        """
        if scale_features:
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        if method == 'univariate':
            # Apply Univariate Feature Selection
            selector = SelectKBest(score_func=chi2, k=k)
            X_new = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_scores = selector.scores_
            print(f"Univariate feature selection scores: {selected_scores}")

        elif method == 'rfe':
            if estimator is None:
                estimator = LogisticRegression(solver='liblinear')  # Default estimator for RFE
            # Apply Recursive Feature Elimination
            selector = RFE(estimator, n_features_to_select=k)
            X_new = selector.fit_transform(X, y)
            selected_indices = selector.support_
            print(f"RFE ranking: {selector.ranking_}")

        elif method == 'importances':
            if estimator is None:
                estimator = RandomForestClassifier()  # Default estimator for feature importances
            # Fit the model to get feature importances
            estimator.fit(X, y)
            importances = estimator.feature_importances_
            selected_indices = np.argsort(importances)[-k:]  # Select top k features
            print(f"Feature importances: {importances}")

        else:
            raise ValueError("Unsupported feature selection method. Choose 'univariate', 'rfe', or 'importances'.")

        # Get feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            selected_features = feature_names[selected_indices]
            return selected_indices, selected_features.tolist()

        return selected_indices

    def fit_xgboost(self, X, y, tune_hyperparameters=True, n_splits=5, random_state=42, n_iter=50, save_model_path=None, **xgb_params):
        """
        Fits an XGBoost model using StratifiedKFold cross-validation and optionally performs hyperparameter tuning.

        Parameters:
            X (np.ndarray): Feature array.
            y (np.ndarray): Label array.
            tune_hyperparameters (bool): If True, perform hyperparameter tuning using RandomizedSearchCV.
            n_splits (int): Number of folds for StratifiedKFold.
            random_state (int): Random seed for reproducibility.
            n_iter (int): Number of iterations for RandomizedSearchCV if tuning hyperparameters.
            save_model_path (str): If provided, saves the best model to this path.
            **xgb_params: Additional parameters to pass to the XGBClassifier if not tuning.

        Returns:
            dict or None: The best hyperparameters found by RandomizedSearchCV if tuning, otherwise None.
            float or None: The best F1 score achieved if tuning, otherwise None.
            XGBClassifier: The trained XGBoost model.
        """
        # Define the parameter grid for XGBoost
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [3, 5, 7, 9, 11],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'min_child_weight': [1, 2, 3, 4, 5]
        }

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        if tune_hyperparameters:
            # Initialize the XGBClassifier model
            model = XGBClassifier(random_state=random_state)

            # Define the scoring function
            scorer = make_scorer(f1_score, average='weighted')  # Use 'binary' for binary classification, 'weighted' for multiclass

            # Initialize RandomizedSearchCV with StratifiedKFold
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=skf,
                random_state=random_state,
                verbose=1,
                n_jobs=-1  # Use all available cores
            )

            # Fit RandomizedSearchCV
            random_search.fit(X, y)

            # Get the best parameters and the best F1 score
            best_params = random_search.best_params_
            best_score = random_search.best_score_

            print(f"Best Parameters: {best_params}")
            print(f"Best F1 Score: {best_score:.4f}")
            # Save the best model if a path is provided
            if save_model_path:
                joblib.dump(random_search.best_estimator_, save_model_path)
                print(f"Best model saved to {save_model_path}")

            # Return the best parameters, best score, and the fitted model
            return best_params, best_score, random_search.best_estimator_

        else:
            # Initialize the XGBClassifier with provided or default parameters
            model = XGBClassifier(random_state=random_state, **xgb_params)

            # Fit the model using all data and StratifiedKFold
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Fit the model
                model.fit(X_train, y_train)

            # Save the model if a path is provided
            if save_model_path:
                joblib.dump(model, save_model_path)
                print(f"Model saved to {save_model_path}")

            print("Model trained with standard hyperparameters.")
            return None, None, model

    def fit_random_forest(self, X, y, tune_hyperparameters=True, n_splits=5, random_state=42, n_iter=50, save_model_path=None, **rf_params):
        """
        Fits a RandomForestClassifier using StratifiedKFold cross-validation and optionally performs hyperparameter tuning.

        Parameters:
            X (np.ndarray): Feature array.
            y (np.ndarray): Label array.
            tune_hyperparameters (bool): If True, perform hyperparameter tuning using RandomizedSearchCV.
            n_splits (int): Number of folds for StratifiedKFold.
            random_state (int): Random seed for reproducibility.
            n_iter (int): Number of iterations for RandomizedSearchCV if tuning hyperparameters.
            save_model_path (str): If provided, saves the best model to this path.
            **rf_params: Additional parameters to pass to the RandomForestClassifier if not tuning.

        Returns:
            dict or None: The best hyperparameters found by RandomizedSearchCV if tuning, otherwise None.
            float or None: The best F1 score achieved if tuning, otherwise None.
            RandomForestClassifier: The trained RandomForest model.
        """
        # Define the parameter grid for RandomForestClassifier
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        if tune_hyperparameters:
            # Initialize the RandomForestClassifier model
            model = RandomForestClassifier(random_state=random_state)

            # Define the scoring function
            scorer = make_scorer(f1_score, average='weighted')  # Use 'binary' for binary classification, 'weighted' for multiclass

            # Initialize RandomizedSearchCV with StratifiedKFold
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=skf,
                random_state=random_state,
                verbose=1,
                n_jobs=-1  # Use all available cores
            )

            # Fit RandomizedSearchCV
            random_search.fit(X, y)

            # Get the best parameters and the best F1 score
            best_params = random_search.best_params_
            best_score = random_search.best_score_

            print(f"Best Parameters: {best_params}")
            print(f"Best F1 Score: {best_score:.4f}")

            # Save the best model if a path is provided
            if save_model_path:
                joblib.dump(random_search.best_estimator_, save_model_path)
                print(f"Best model saved to {save_model_path}")

            # Return the best parameters, best score, and the fitted model
            return best_params, best_score, random_search.best_estimator_

        else:
            # Initialize the RandomForestClassifier with provided or default parameters
            model = RandomForestClassifier(random_state=random_state, **rf_params)

            # Fit the model using all data and StratifiedKFold
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Fit the model
                model.fit(X_train, y_train)

            print("Model trained with standard hyperparameters.")

            # Save the model if a path is provided
            if save_model_path:
                joblib.dump(model, save_model_path)
                print(f"Model saved to {save_model_path}")

            return None, None, model

    def fit_perceptron(self, X, y, tune_hyperparameters=True, n_splits=5, random_state=42, n_iter=50, save_model_path=None, **mlp_params):
        """
        Fits an MLPClassifier using StratifiedKFold cross-validation and optionally performs hyperparameter tuning.

        Parameters:
            X (np.ndarray): Feature array.
            y (np.ndarray): Label array.
            tune_hyperparameters (bool): If True, perform hyperparameter tuning using RandomizedSearchCV.
            n_splits (int): Number of folds for StratifiedKFold.
            random_state (int): Random seed for reproducibility.
            n_iter (int): Number of iterations for RandomizedSearchCV if tuning hyperparameters.
            save_model_path (str): If provided, saves the best model to this path.
            **mlp_params: Additional parameters to pass to the MLPClassifier if not tuning.

        Returns:
            dict or None: The best hyperparameters found by RandomizedSearchCV if tuning, otherwise None.
            float or None: The best F1 score achieved if tuning, otherwise None.
            MLPClassifier: The trained MLP model.
        """
        # Define the parameter grid for MLPClassifier
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'batch_size': ['auto', 10, 20, 40, 60],
            'momentum': [0.9, 0.8, 0.7, 0.6, 0.5]  # Only applicable for 'sgd' solver
        }

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        if tune_hyperparameters:
            # Initialize the MLPClassifier model
            model = MLPClassifier(random_state=random_state)

            # Define the scoring function
            scorer = make_scorer(f1_score, average='weighted')  # Use 'binary' for binary classification, 'weighted' for multiclass

            # Initialize RandomizedSearchCV with StratifiedKFold
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=skf,
                random_state=random_state,
                verbose=1,
                n_jobs=-1  # Use all available cores
            )

            # Fit RandomizedSearchCV
            random_search.fit(X, y)

            # Get the best parameters and the best F1 score
            best_params = random_search.best_params_
            best_score = random_search.best_score_

            print(f"Best Parameters: {best_params}")
            print(f"Best F1 Score: {best_score:.4f}")

            # Save the best model if a path is provided
            if save_model_path:
                joblib.dump(random_search.best_estimator_, save_model_path)
                print(f"Best model saved to {save_model_path}")

            # Return the best parameters, best score, and the fitted model
            return best_params, best_score, random_search.best_estimator_

        else:
            # Initialize the MLPClassifier with provided or default parameters
            model = MLPClassifier(random_state=random_state, **mlp_params)

            # Fit the model using all data and StratifiedKFold
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Fit the model
                model.fit(X_train, y_train)

            print("Model trained with standard hyperparameters.")

            # Save the model if a path is provided
            if save_model_path:
                joblib.dump(model, save_model_path)
                print(f"Model saved to {save_model_path}")

            return None, None, model

    def evaluate_model(self, model, X_test, y_test, savename=None):
        """
        Evaluates the performance of a trained model on a test set.

        Parameters:
            model (XGBClassifier or any sklearn-compatible model): The trained model to be evaluated.
            X_test (np.ndarray): The feature array for the test set.
            y_test (np.ndarray): The label array for the test set.

        Returns:
            dict: A dictionary containing various evaluation metrics.
        """
        # Predict the labels for the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass, 'binary' for binary classification
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Print a classification report
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Create a confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        if savename:
            plt.savefig(os.path.join(figures_path, savename), bbox_inches='tight', dpi=300)
        else:
            plt.show()

        # Return metrics as a dictionary
        metrics = {
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Confusion Matrix': conf_matrix
        }

        return metrics

# Argument Definition 
parser = argparse.ArgumentParser(
    prog = "Machine Learning Model Prediciton",
    description = "This script is used to predict the finalist, playoff partecipants and winner (or mvp)"
)

parser.add_argument("--dataset", choices=["teams", "players"], help = "which dataset to use for the analysis")
parser.add_argument("--label_column", choices=["Playoff", "Winners", "Finalist", "MVP"], help = "which label to use for the prediction")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    label_column = args.label_column

    # path definition
    script_path = os.path.dirname(os.path.abspath("ml_prediction.py"))
    data_path = os.path.join(script_path, "../data/")
    output_path = os.path.join(script_path, "../output/")
    figures_path = os.path.join(output_path, "figures/")

    if dataset == "teams":
        data_file = os.path.join(data_path, "teams_stats_2003-2004_2023-2024.csv")
    if dataset == "players":
        data_file = os.path.join(data_path, "players_stats_2003-2004_2023-2024.csv")

    # class initialization
    print("\nClass Initialization")
    mlp = MLPredictor(data_file, output_path, figures_path, "teams")

    # read the csv file as pandas dataframe
    print(f"Read the {data_file} dataset")
    df = mlp.read_data()

    # make features and label numpy array
    print("Creation of the features and label NumPy array")
    X, y = mlp.dataframe_to_numpy(df, label_column, features_to_drop=["Year", "MPG", "FGM"])

    # train and test split
    print("Train and Test split")
    X_train, X_test, y_train, y_test = mlp.split_dataset(X, y, test_size=0.3, shuffle=True, random_state=42)

    # fit the model
    print("Fit the ML model")
    _, _,model = mlp.fit_xgboost(X_train, y_train, tune_hyperparameters=False, n_splits=5, random_state=42, n_iter=50)
    #_,_,model = mlp.fit_random_forest(X_train, y_train, tune_hyperparameters=False, n_splits=5, random_state=42, n_iter=50, save_model_path=None)
    #_,_,model = mlp.fit_perceptron(X_train, y_train, tune_hyperparameters=False, n_splits=5, random_state=42, n_iter=100, save_model_path=None)

    # evaluate the model
    print("Evaluate the best model")
    mlp.evaluate_model(model, X_test, y_test, savename="prova.png")

