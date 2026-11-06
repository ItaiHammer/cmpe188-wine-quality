# Functions for preparing wine-quality data before model training.

# Imports for preprocessing
import os, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Load and check the data
df = pd.read_csv("data/10_wine_quality_train.csv")
print(df.head())
print(df.info())
print(df.describe())

# Define the target variable and features
target = "good_quality"
x = df.drop(columns=[target])
y = df[target]

# Split the data into training and testing sets, 20% for testing, stratified by the target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

