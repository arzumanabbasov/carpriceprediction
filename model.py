import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostRegressor
import joblib

# Load the dataset
df = pd.read_csv('car_prediction_data.csv')

# Drop the unwanted columns
df.drop(['Car_Name'], axis=1, inplace=True)

# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Calculate the age of the car
df['Age'] = 2023 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

# Split the dataset into features (X) and target (y)
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature selection using ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X_train, y_train)

# Plot feature importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# Hyperparameter tuning using RandomizedSearchCV and CatBoostRegressor
cb = CatBoostRegressor()
grid = {
    'learning_rate': [0.03, 0.1],
    'depth': [4, 6, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}
cb_random = RandomizedSearchCV(estimator=cb, param_distributions=grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)
cb_random.fit(X_train, y_train)

# Evaluate the model on the test set
predictions = cb_random.predict(X_test)
sns.displot(y_test-predictions)
plt.scatter(y_test, predictions)

# Print the best score and parameters
print("Best score:", cb.best_score_)

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

# Save the model using joblib
joblib.dump(cb_random.best_estimator_, 'car_price_model.joblib')

# Load the model from the saved file
loaded_model = joblib.load('car_price_model.joblib')
