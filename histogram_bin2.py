#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:47:19 2023

@author: saiful
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% UK
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_UK2.csv'
df_UK = pd.read_csv(file_path)
column_names = df_UK.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = df_UK['UK'].min()
max_value = df_UK['UK'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_UK = np.histogram(df_UK['UK'], bins=num_bins, range=(min_value, max_value))
print("Bin bin_edges_UK:", bin_edges_UK)

# Plot the histogram
plt.hist(df_UK['UK'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of UK column')
plt.xlabel('UK values')
plt.ylabel('Frequency')
plt.show()


#%% DE

file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv'
df_DE = pd.read_csv(file_path)
column_names = df_DE.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = df_DE['DE'].min()
max_value = df_DE['DE'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_DE = np.histogram(df_DE['DE'], bins=num_bins, range=(min_value, max_value))
print("Bin bin_edges_DE:", bin_edges_DE)

# Plot the histogram
plt.hist(df_DE['DE'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of DE column')
plt.xlabel('DE values')
plt.ylabel('Frequency')
plt.show()

#%% FR
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_FR2.csv'
df_FR = pd.read_csv(file_path)
column_names = df_FR.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = df_FR['FR'].min()
max_value = df_FR['FR'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_FR = np.histogram(df_FR['FR'], bins=num_bins, range=(min_value, max_value))
print("Bin bin_edges_FR:", bin_edges_FR)

# Plot the histogram
plt.hist(df_FR['FR'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of FR column')
plt.xlabel('FR values')
plt.ylabel('Frequency')
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# File paths
file_paths = [
    '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_UK2.csv',
    '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv',
    '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_FR2.csv',
]

# Read dataframes
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Concatenate all data to find the overall min and max
# all_data = pd.concat([df['UK'] for df in dfs] + [df['DE'] for df in dfs] + [df['FR'] for df in dfs])
all_data = pd.concat([df_UK['UK'], df_DE['DE'], df_FR['FR']])

# Calculate min and max values
min_value_of_all = all_data.min()   # 1e-05
max_value_of_all = df_UK['UK'].max()  # 0.98527


# =============================================================================
# #%% AFTER FIXING THE RANGE
# =============================================================================
#%% UK
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_UK2.csv'
df_UK = pd.read_csv(file_path)
column_names = df_UK.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
min_value = min_value_of_all
max_value = max_value_of_all

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_UK = np.histogram(df_UK['UK'], bins=num_bins, range=(min_value, max_value))
print("flag 1.21 Bin bin_edges_UK AFTER FIXING RANGE:", bin_edges_UK)

# Plot the histogram
plt.hist(df_UK['UK'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of UK column AFTER FIXING RANGE')
plt.xlabel('UK values')
plt.ylabel('Frequency')
plt.show()


#%% DE

file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv'
df_DE = pd.read_csv(file_path)
column_names = df_DE.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
# min_value = df_DE['DE'].min()
# max_value = df_DE['DE'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_DE = np.histogram(df_DE['DE'], bins=num_bins, range=(min_value, max_value))
print(" flag 1.21 Bin bin_edges_DE AFTER FIXING RANGE:", bin_edges_DE)

# Plot the histogram
plt.hist(df_DE['DE'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of DE column AFTER FIXING RANGE')
plt.xlabel('DE values')
plt.ylabel('Frequency')
plt.show()
#%% DE

file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_DE2.csv'
df_DE = pd.read_csv(file_path)
column_names = df_DE.columns
print("Column Names:")
print(column_names)
# colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'purple', 'yellow']
# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
# min_value = df_DE['DE'].min()
# max_value = df_DE['DE'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_DE = np.histogram(df_DE['DE'], bins=num_bins, range=(min_value, max_value))
print(" flag 1.21 Bin bin_edges_DE AFTER FIXING RANGE:", bin_edges_DE)

# Plot the histogram
plt.hist(df_DE['DE'], bins=num_bins, range=(min_value, max_value), edgecolor='black', color='salmon')
# plt.title('Histogram of DE column AFTER FIXING RANGE')
plt.xlabel('Generated Wind Power (Normalized)')
plt.ylabel('Frequency')
plt.savefig('germany windpower histogram', dpi = 300)
plt.show()

#%% FR
file_path = '/home/saiful/energy_theft/wind power/data/europe/weather_and_power_merged/merged_df_FR2.csv'
df_FR = pd.read_csv(file_path)
column_names = df_FR.columns
print("Column Names:")
print(column_names)

# df = pd.DataFrame({'DE': np.random.randint(0, 100, 47000)})

# Calculate min and max values of the 'DE' column
# min_value = df_FR['FR'].min()
# max_value = df_FR['FR'].max()

# Create histogram with 10 bins
num_bins = 6
hist, bin_edges_FR = np.histogram(df_FR['FR'], bins=num_bins, range=(min_value, max_value))
print(" flag 1.21 Bin bin_edges_FR AFTER FIXING RANGe:", bin_edges_FR)

# Plot the histogram
plt.hist(df_FR['FR'], bins=num_bins, range=(min_value, max_value), edgecolor='black')
plt.title('Histogram of FR column AFTER FIXING RANGe')
plt.xlabel('FR values')
plt.ylabel('Frequency')
plt.show()
