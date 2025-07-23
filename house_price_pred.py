import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Dataset loading
df = pd.read_csv("train.csv")
#print shape of hosue
print("Dataset shape:", df.shape)

#for price prediction perspective, both types of bathrooms contribute to the house’s value but not equally.so combining both
df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

#relevant features & drop missing values
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
df = df[features + ['SalePrice']].dropna()

#define input features (X) & target (Y)
X =df[features]
y =df['SalePrice']

#Split into training and training sets(80% - train, 20% - test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Linear Regression model & make predictions on test set
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 score = r2_score(y_test, y_pred)

#Print outputs
print("\n=== Model Evaluation ===")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R2 Score:", round(r2, 4))
print("\n=== Model Coefficients ===")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")
print("Intercept:", model.intercept_)


# Predict price for a custom input, Example: 2000 sqft, 3 bedrooms, 2 full + 1 half bath
custom_input = pd.DataFrame([[2000, 3, 2 + 0.5 * 1]], columns=features)
custom_prediction = model.predict(custom_input)[0]
print("\nPredicted Price for custom input:", round(custom_prediction, 2))











