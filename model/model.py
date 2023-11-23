from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd

# Import Preprocessor class
from pre_processing import Preprocessor

# Initialize the preprocessor with the dataset
preprocessor = Preprocessor('dataset/AI_ML_YT_Videos.csv')


# Preprocess the dataset
processed_data = preprocessor.data 
Y = processed_data['Views']  

# Generate features using the preprocessor
X = pd.DataFrame()
for i, row in processed_data.iterrows():
    processed_row = preprocessor.transform_input(row['Title'], row['Channel'], row['PublishedDate'])
    X = pd.concat([X, processed_row], ignore_index=True)


# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = random_forest_model.predict(X_test)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print(f"Root Mean Squared Error: {rmse}")

# Save the trained model and preprocessor
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(random_forest_model, file)
