from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Profiling function for the numerical variables
def numerical_profiling(df, columns):
    results = {}
    for col in columns:
        col_data = df[col]
                
        # Shapiro-Wilk test for normality
        stat, p_value = shapiro(col_data)
        normality = "Normal" if p_value > 0.05 else "Not Normal"
                
        # Basic statistics
        col_mean = col_data.mean()
        col_max = col_data.max()
        col_min = col_data.min()
        col_std = col_data.std()
        col_variance = col_data.var()
        missing_values = col_data.isnull().sum()
        null_ratio = missing_values / len(col_data) # Ratio of nulls to total rows

        # IQR and Outlier Detection
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
        
        # Plotting boxplot
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=col_data)
        plt.title(f"Boxplot of {col}")
        plt.show()

        # Store results
        results[col] = {
            "missing_values": missing_values,
            "null_ratio": null_ratio,
            "mean": col_mean,
            "max": col_max,
            "min": col_min,
            "std_dev": col_std,
            "variance": col_variance,
            "normality (Shapiro-Wilk)": normality,
            "outliers (IQR)": list(set(outliers)) # List for easy viewing
        }
    # Return dictionary of results
    return pd.DataFrame(results).T # Transpose for readability


# Profinling functions for categorical variables, which only have a single category in a row
def categorical_profiling(df, columns):
    results = {}
     
    for col in columns:
        col_data = df[col].dropna() # Drop NaNs to calculate frequencies correctly
        
        # Frequency of each category
        freq_dist = col_data.value_counts()
        
        # Calculation of specific values
        unique_count = col_data.nunique()
        null_count = df[col].isnull().sum() # Total nulls in the original column
       
        # Dominant and least frequent categories
        dominant_category = freq_dist.idxmax() if not freq_dist.empty else None
        dominant_freq = freq_dist.max() if not freq_dist.empty else None
        least_freq_category = freq_dist[freq_dist == freq_dist.min()].index.tolist() if not freq_dist.empty else []
        least_freq = freq_dist.min() if not freq_dist.empty else None
        null_ratio = null_count / len(df) # Ratio of nulls to total rows


        # Frequency distribution of top 10 categories
        top_n = freq_dist.head(10)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_n.index, y=top_n.values, palette="viridis")
        
        plt.title(f"Frequency Distribution of Top 10 {col} Values")
        plt.xticks(rotation=45)
        plt.show()
        
        # Storing results
        results[col] = {
            "unique_categories": unique_count,
            "null_values": null_count,
            "dominant_category": dominant_category,
            "dominant_frequency": dominant_freq,
            "least_frequent_category": least_freq_category,
            "least_frequency": least_freq,
            "null_ratio" : null_ratio
        }
    # Return the results as a DataFrame for easy visualization
    return pd.DataFrame(results).T # Transpose for readability


# Profiling function for multi-category variables, which have multiple categories in a row separated by commas
def categorical_profiling_multi(df, columns):
    results = {}

    for col in columns:
        # Split multi-category values, flatten into a single list of categories for the column
        flat_col_data = df[col].dropna().apply(lambda x: x.split(', ')).explode()

        # Frequency of each category
        freq_dist = flat_col_data.value_counts()

        # Calculation of specific values
        unique_count = flat_col_data.nunique()
        null_count = df[col].isnull().sum()     # Total nulls in the original column
        null_ratio = null_count / len(df[col])  # Ratio of nulls to total rows

        # Dominant and least frequent categories
        dominant_category = freq_dist.idxmax() if not freq_dist.empty else None
        dominant_freq = freq_dist.max() if not freq_dist.empty else None
        least_freq_category = freq_dist[freq_dist == freq_dist.min()].index.tolist() if not freq_dist.empty else []
        least_freq = freq_dist.min() if not freq_dist.empty else None

        # Frequency distribution of top 10 categories
        top_n = freq_dist.head(10)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_n.index, y=top_n.values, palette="viridis")
        plt.title(f"Frequency Distribution of Top 10 {col} Values")
        plt.xticks(rotation=45)
        plt.show()

        # Storing results
        results[col] = {
            "unique_categories": unique_count,
            "null_values": null_count,
            "null_ratio": null_ratio,
            "dominant_category": dominant_category,
            "dominant_frequency": dominant_freq,
            "least_frequent_category": least_freq_category,
            "least_frequency": least_freq
        }

    # Return the results as a DataFrame for easy visualization
    return pd.DataFrame(results).T # Transpose for readability