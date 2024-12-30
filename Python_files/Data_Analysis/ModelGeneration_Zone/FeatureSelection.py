import duckdb
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression,r_regression
import numpy as np
import pandas as pd


def varianceThreshold(X):
    try:
        sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
        sel.fit_transform(X)
        print(f'Variance Threshold Variablles: {sel.get_feature_names_out()}')
        return (True, ', '.join(sel.get_feature_names_out().tolist()))
    except Exception as e:
        print(f"An error occurred: {e}")
        return (False,)

def rRegression(X, y):
    try:
        r_statistic = r_regression(X, y)
        r_table = pd.DataFrame({
            'Variable': X.columns.tolist(),
            'Correlation Coefficient (r)': r_statistic
        })
        correlated_variables = r_table[r_table['Correlation Coefficient (r)'] != 0]['Variable']
        print(f'Correlated Variables: {list(correlated_variables)}')
        print(correlated_variables)
        return (True, ', '.join(list(correlated_variables)))
    except Exception as e:
        print(f"An error occurred: {e}")
        return (False,)

def fRegression(X, y):
    try:
        f_statistic, p_values = f_regression(X, y)
        significant_mask = p_values < 0.05
        significant_features = X.columns[significant_mask]
        print(f"Significant features: {list(significant_features)}")
        print(significant_features)
        return (True, ', '.join(list(significant_features)))
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return (False,)