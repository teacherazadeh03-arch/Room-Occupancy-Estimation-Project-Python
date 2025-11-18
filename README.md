# Room-Occupancy-Estimation-Project-Python
This project uses the Room Occupancy dataset from the UCI Machine Learning Repository  to predict the number of occupants in a room based on sensor data. The dataset includes measurements such as temperature, light, sound, CO2 levels, and PIR sensor readings.

The project applies multiple machine learning models to classify room occupancy, including:
K-Nearest Neighbours (KNN)
Logistic Regression
Decision Tree Classifier
Random Forest Classifier

Dataset Details

Rows: 10,129
Columns: 18 features + target

Features:

Date, Time (dropped during preprocessing)
S1_Temp, S2_Temp, S3_Temp, S4_Temp
S1_Light, S2_Light, S3_Light, S4_Light
S1_Sound, S2_Sound, S3_Sound, S4_Sound
S5_CO2, S5_CO2_Slope
S6_PIR, S7_PIR

Target: Room_Occupancy_Count (0 = empty, 1–3 = occupied)

Features Used

Only numeric sensor readings are used for modelling. Non-numeric columns (Date and Time) are removed because machine learning models require numeric input.

Preprocessing Steps
Drop non-numeric columns (Date, Time).
Handle missing values (if any) using mean imputation.
Scale numeric features using StandardScaler for models sensitive to feature magnitude (KNN, Logistic Regression).
Train/Test split

80% training, 20% testing

Stratified split to preserve class distribution
Models Applied

K-Nearest Neighbours (KNN)
Works well with numeric data
Scaled features improve distance calculations

Logistic Regression
Multiclass classification
Uses scaled features

Decision Tree Classifier
Handles unscaled numeric data naturally
Provides feature importance

Random Forest Classifier
Ensemble of Decision Trees
Provides more robust predictions and feature importance
