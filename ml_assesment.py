#-------------------------------
# Penguin Body Mass Prediction
#-------------------------------

#-------------------------------
# Exploratory Data Analysis (EDA)
#-------------------------------

import pandas as pd

# Load the dataset
df = pd.read_csv('modified_penguins.csv')

# Display the first 5 rows
print(df.head()) # Loaded correctly

# Check the dataset size (rows and columns)
print("Dataset size (Rows, Columns):")
print(df.shape)

# Check column names and data types
print("Column names and datatypes:")
print(df.dtypes)

# results
#Column names and datatypes:
#studyName                  str
#Region                     str
#Island                     str
#Stage                      str
#Individual ID              str
#Clutch Completion          str
#Date Egg                   str
#Culmen Length (mm)     float64
#Culmen Depth (mm)      float64
#Flipper Length (mm)    float64
#Body Mass (g)            int64
#Sex                        str
#Delta 15 N (o/oo)      float64
#Delta 13 C (o/oo)      float64
#Comments                   str

# CHANGE STRINGS TO NUMBERS IN FUTURE

# Check dataset for missing values
print("Missing values per column:")
print(df.isnull().sum())

#results
# Culmen Length - 4
# Culmen Depth - 2
# Flipper Length - 2
# Sex - 23
# Delta 15 N - 14
# Delta 13 C - 13
# Comments - 374

# Check description of my target, body mass
print("Body Mass Description:")
print(df['Body Mass (g)'].describe())

#---------------------------------------
# Pre-processing - Clean the data
#---------------------------------------
# Drop columns that are irrelevant to target
# studyName, Individual ID, Date Egg and Comments
df = df[['Island', 'Culmen Length (mm)', 'Culmen Depth (mm)'
    , 'Flipper Length (mm)', 'Sex',
      'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 'Body Mass (g)']]

print("Remaining columns:")
print(df.columns.tolist())
print("Dataset size after changes (Rows, Columns):")
print(df.shape)

# New dataset size (Rows, Columns)
# (444, 8)

# Clean invalid Sex values
df = df[df['Sex'].isin(['MALE', 'FEMALE'])]

print("Dataset size after removing invalid Sex values:")
print(df.shape)

# New dataset size (Rows, Columns)
# (420, 8)

# Drop any remaining rows that have missing values
df = df.dropna()

# Check missing values after clean
print("Missing values after cleaning:")
print(df.isnull().sum())

# result = 0

# Check final dataset size
print("Final dataset size:")
print(df.shape)

# Final dataset size
# (405, 8)

# Convert strings to numbers
# Encode the variables
from sklearn.preprocessing import LabelEncoder

# Create the encoders for each column
le_island = LabelEncoder()
le_sex = LabelEncoder()

# Encode Island (Biscoe = 0, Dream = 1, Torgersen = 2)
df['Island'] = le_island.fit_transform(df['Island'])

# Encode Sex (FEMALE = 0, MALE = 1)
df['Sex'] = le_sex.fit_transform(df['Sex'])

# Check encoding has worked
print("First 5 rows after encoding")
print(df.head())

#----------------------
# Visualisations
#----------------------
import matplotlib.pyplot as plt

# Check to see how spread out the values are
# PLot histogram of Body Mass
plt.figure(figsize=(8,6))
plt.hist(df['Body Mass (g)'], bins=50)
plt.title('Distribution of Body Mass')
plt.xlabel('Body Mass (g)')
plt.ylabel('Count')
plt.savefig('body_mass_distribution.png')
plt.show()

# Results
# Ranges from 2500g - 6300g

# Check Body Mass by Sex
# Boxplot chart to show body mass distribution by sex
# Do males and females have noticeably different body mass
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='Body Mass (g)', data=df,
            hue='Sex', palette=['salmon', 'blue'], legend=False)
plt.xticks([0, 1], ['FEMALE', 'MALE'])
plt.title('Body Mass by Sex')
plt.xlabel('Sex')
plt.ylabel('Body Mass (g)')
plt.tight_layout()
plt.savefig('body_mass_by_sex.png')
plt.show()

# Results
# Both male and females have very similar body mass distribution

# Heatmap chart to show relationships of all values
# Which features have the strongest relationship with Body Mass?

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# Results
# Culmen Length shows a strong relationship with body mass
# Longer beaks tend to mean lighter penguins

#------------------------------------
# Split data for features and target
#------------------------------------

# X is all columns except Body Mass for inputs
X = df.drop(columns=['Body Mass (g)'])

# y is Body Mass column for prediction target
y = df['Body Mass (g)']

print("Features (X):")
print(X.head())
print("Target (y): Body Mass (g)")
print(f"Number of samples: {len(y)}")

# results
# X has island, culmen length, culmen depth, flipper length, sex, delta 15 n and delta 13 c
# y has 405 Body Mass values to predict

#--------------------------------------–––––
# Split data into training and testing sets
#–––––––––––––––––––––––––––––––––––––––––––

from sklearn.model_selection import train_test_split

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# results
# training samples - 324
# testing samples - 81

#-----------------
# Decision Tree
#-----------------
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(max_depth=2, random_state=42)

# Train the model with training data
dt_model.fit(X_train, y_train)

print(f"Tree depth: {dt_model.get_depth()}")

#----------------------------------
# Make Predictions with the model
#----------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Performance:")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} grams")

# results
# RMSE = 494g - my models predictions are about 494 grams off from the actual body mass
# MSE = 244176.14

#--------------------------------
# Actual vs Predicted plot
#--------------------------------

# Use scatter plot chart to show each penguin as a dot and where they sit compared to where the model predicted
# The closer the dots are to the red line the better the model is performing

# Plot actual vs predicted body mass values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Body Mass')
plt.xlabel('Actual Body Mass (g)')
plt.ylabel('Predicted Body Mass (g)')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# results
# The model is learning, it does better with higher body masses (5000-6000g) - dots cluster close to line
# It is struggling more at lower body masses(2500-3500g) - dots are more scattered

#-------------------------
# Hyperparameter tuning
#-------------------------

# I am changing max_depth values to find the best performance
# max_depth = 3
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Model Performance:")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} grams")

# results
# down to 485g off from actual body mass

# max_depth = 7
dt_model = DecisionTreeRegressor(max_depth=7, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Model Performance:")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} grams")

# results
# up to 507g off from actual body mass
# when I increase the depth the model gets worse
# when i decrease it gets better

# max_depth = 2
dt_model = DecisionTreeRegressor(max_depth=2, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Model Performance:")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} grams")


# results
# RMSE - 455.99 grams
# max _depth = 2 is the best performing

#-----------------------------------------------------
# Final model selected based on hyperparameter tuning
#-----------------------------------------------------

dt_model = DecisionTreeRegressor(max_depth=2, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("Final model trained with max_depth = 2")
print(f"Final RMSE {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} grams")

# At this point i changed the hyperparameter for the chart

# R2 Score, overall performance score
# 1.0 = perfect predictions
# 0.0 = no better than guessing
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2: .2f}")

# result
# R2 score: 0.76

#-------------------------------
# Penguin Body Mass Predictor
#-------------------------------

def predict_body_mass():
    print("Penguin Body Mass Predictor")
    print("Enter the following measurements:")

    culmen_length = float(input("Enter the culmen length (mm) e.g. 46.5: "))
    culmen_depth = float(input("Enter the culmen depth (mm) e.g. 17.3: "))
    flipper_length = float(input("Enter the flipper length (mm) e.g. 195: "))
    sex = input("Sex (MALE / FEMALE):").strip().upper()
    island = input("Island (Biscoe / Dream / Torgersen):").strip().title()
    delta_15n = float(input("Delta 15 N (o/oo) e.g. 8.5: "))
    delta_13c = float(input("Delta 13 c (o/oo) e.g. -25.5: "))

    # Encode sex and island inputs
    sex_encoded = le_sex.transform([sex])[0]
    island_encoded = le_island.transform([island])[0]

    #Create a dataframe from the input
    input_data_sample = pd.DataFrame([[island_encoded, culmen_length, culmen_depth, flipper_length, sex_encoded, delta_15n, delta_13c]], columns=X.columns)

    # Make predictions
    prediction = dt_model.predict(input_data_sample)

    print(f"Predicted Body Mass: {prediction[0]:.0f} g")

# Run the predictor
predict_body_mass()