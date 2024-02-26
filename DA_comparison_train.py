#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:42:06 2024

@author: saiful
"""

import pandas as pd

#%% Germany
df_DE_train_acc_from_scratch300 = pd.read_excel("/home/saiful/energy_theft/wind power/DE_train_acc_from_scratch300.xlsx")
print(df_DE_train_acc_from_scratch300.head())
df_DE1 = df_DE_train_acc_from_scratch300[['DE train from scratch']]

DE_traintest_acc_after_domain_adaptation_from_FR = pd.read_excel("/home/saiful/energy_theft/wind power/DE_traintest_acc_after_domain_adaptation_from_FR300.xlsx")
df_DE2 = DE_traintest_acc_after_domain_adaptation_from_FR[['DE train after Domain Adaptation']]
df_DE2 = df_DE2.rename(columns={'DE train after Domain Adaptation': 'DE train after Domain Adaptation from FR'})

DE_traintest_acc_after_domain_adaptation_from_UK = pd.read_excel("/home/saiful/energy_theft/wind power/DE_traintest_acc_after_domain_adaptation_from_UK300.xlsx")
df_DE3 = DE_traintest_acc_after_domain_adaptation_from_UK[['DE train after Domain Adaptation']]
df_DE3 = df_DE3.rename(columns={'DE train after Domain Adaptation': 'DE train after Domain Adaptation from UK'})

merged_df = pd.concat([df_DE1, df_DE2, df_DE3], ignore_index=False, axis=1)
print(merged_df.columns)

##
import matplotlib.pyplot as plt

# Plotting the columns
plt.figure(figsize=(10, 6))  # Set the figure size
for column in merged_df.columns:
    if column == 'DE train from scratch':
        plt.plot(merged_df.index, merged_df[column]* 100, label='From scratch')
    elif column == 'DE train after Domain Adaptation from FR':
        plt.plot(merged_df.index, merged_df[column]* 100, label='After Domain Adaptation from France')
    elif column == 'DE train after Domain Adaptation from UK':
        plt.plot(merged_df.index, merged_df[column]* 100, label='After Domain Adaptation from UK')

# Adding labels and title
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Train accuracy', fontsize=14)
plt.title('Germany', fontsize=16)

# Adding legend
plt.legend()
plt.savefig('germany_train_accuracy with domain adaptation.png', dpi=300)

# Show plot
plt.show()

#%%
#%% UK
df_UK_train_acc_from_scratch300 = pd.read_excel("/home/saiful/energy_theft/wind power/UK_train_acc_from_scratch300.xlsx")
print(df_UK_train_acc_from_scratch300.head())
df_UK1 = df_UK_train_acc_from_scratch300[['UK train from scratch']]

UK_traintest_acc_after_domain_adaptation_from_FR = pd.read_excel("/home/saiful/energy_theft/wind power/UK_traintest_acc_after_domain_adaptation_from_FR300.xlsx")
df_UK2 = UK_traintest_acc_after_domain_adaptation_from_FR[['UK train after Domain Adaptation']]
df_UK2 = df_UK2.rename(columns={'UK train after Domain Adaptation': 'UK train after Domain Adaptation from FR'})

UK_traintest_acc_after_domain_adaptation_from_DE = pd.read_excel("/home/saiful/energy_theft/wind power/UK_traintest_acc_after_domain_adaptation_from_DE300.xlsx")
df_UK3 = UK_traintest_acc_after_domain_adaptation_from_DE[['UK train after Domain Adaptation']]
df_UK3 = df_UK3.rename(columns={'UK train after Domain Adaptation': 'UK train after Domain Adaptation from DE'})

merged_df = pd.concat([df_UK1, df_UK2, df_UK3], ignore_index=False, axis=1)
print(merged_df.columns)

##
import matplotlib.pyplot as plt

# Plotting the columns
plt.figure(figsize=(10, 6))  # Set the figure size
for column in merged_df.columns:
    if column == 'UK train from scratch':
        plt.plot(merged_df.index, merged_df[column]* 100, label='From scratch')
    elif column == 'UK train after Domain Adaptation from FR':
        plt.plot(merged_df.index, merged_df[column]* 100, label='After Domain Adaptation from France')
    elif column == 'UK train after Domain Adaptation from DE':
        plt.plot(merged_df.index, merged_df[column]* 100, label='After Domain Adaptation from Germany')

# Adding labels and title
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Train accuracy', fontsize=14)
plt.title('United Kingdom', fontsize=16)

# Adding legend
plt.legend()
plt.savefig('UK_train_accuracy with domain adaptation.png', dpi=300)

# Show plot
plt.show()
#%% FR
df_FR_train_acc_from_scratch300 = pd.read_excel("/home/saiful/energy_theft/wind power/FR_train_acc_from_scratch300.xlsx")
print(df_FR_train_acc_from_scratch300.head())
df_FR1 = df_FR_train_acc_from_scratch300[['FR train from scratch']]

FR_traintest_acc_after_domain_adaptation_from_UK = pd.read_excel("/home/saiful/energy_theft/wind power/FR_traintest_acc_after_domain_adaptation_from_UK300.xlsx")
df_FR2 = FR_traintest_acc_after_domain_adaptation_from_UK[['FR train after Domain Adaptation']]
df_FR2 = df_FR2.rename(columns={'FR train after Domain Adaptation': 'FR train after Domain Adaptation from UK'})

FR_traintest_acc_after_domain_adaptation_from_DE = pd.read_excel("/home/saiful/energy_theft/wind power/FR_traintest_acc_after_domain_adaptation_from_DE300.xlsx")
df_FR3 = FR_traintest_acc_after_domain_adaptation_from_DE[['FR train after Domain Adaptation']]
df_FR3 = df_FR3.rename(columns={'FR train after Domain Adaptation': 'FR train after Domain Adaptation from DE'})

merged_df = pd.concat([df_FR1, df_FR2, df_FR3], ignore_index=False, axis=1)
print(merged_df.columns)

##
import matplotlib.pyplot as plt

# Plotting the columns
plt.figure(figsize=(10, 6))  # Set the figure size
for column in merged_df.columns:
    if column == 'FR train from scratch':
        plt.plot(merged_df.index, merged_df[column]* 100, label='From scratch')
    elif column == 'FR train after Domain Adaptation from UK':
        plt.plot(merged_df.index, merged_df[column]* 100, label='After Domain Adaptation from UK')
    elif column == 'FR train after Domain Adaptation from DE':
        plt.plot(merged_df.index, merged_df[column]* 100, label='After Domain Adaptation from Germany')

# Adding labels and title
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Train accuracy', fontsize=14)
plt.title('France', fontsize=16)

# Adding legend
plt.legend()
plt.savefig('france_train_accuracy with domain adaptation.png', dpi=300)

# Show plot
plt.show()