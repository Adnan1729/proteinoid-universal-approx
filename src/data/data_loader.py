# data_loader.py (src/data/data_loader.py)
import pandas as pd
import numpy as np

def load_spike_data(file_path):
    """
    Load and process spike data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing spike data
        
    Returns:
        list: List of processed dataframes for each dataset
    """
    spike_data = pd.read_csv(file_path)
    return process_all_datasets(spike_data)

def process_dataset(df, column_name):
    """
    Process individual dataset from the spike data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Name of the column to process
        
    Returns:
        pd.DataFrame: Processed dataframe with Time and Spike columns
    """
    data = df[[column_name, df.columns[df.columns.get_loc(column_name) + 1]]].copy()
    data.columns = ['Time', 'Spike']
    data = data.iloc[1:]  # Remove the header row
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
    data['Spike'] = pd.to_numeric(data['Spike'], errors='coerce')
    data = data.dropna()
    data['Dataset'] = column_name
    return data

def process_all_datasets(spike_data):
    """
    Process all datasets from the spike data.
    
    Args:
        spike_data (pd.DataFrame): Input dataframe containing all datasets
        
    Returns:
        list: List of processed dataframes
    """
    dataframes = []
    for column in ['Data1', 'Data2', 'Data3', 'Data4', 'Data5']:
        if column in spike_data.columns:
            df = process_dataset(spike_data, column)
            dataframes.append(df)
    return dataframes
