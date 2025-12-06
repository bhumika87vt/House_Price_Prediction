import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

print("Starting script...")

# 1. Load data
df = pd.read_csv("train.csv")
print("Data loaded. Shape:", df.shape)

# 2. Encode 'Neighborhood' (location) as numbers
le = LabelEncoder()
df["NeighborhoodIndex"] = le.fit_transform(df["Neighborhood"])

# 3. Select features and target
X = df[["GrLivArea", "BedroomAbvGr", "FullBath", "TotRmsAbvGrd", "NeighborhoodIndex"]]
y = df["SalePrice"]

# 4. Train–test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate with RMSE
y_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
print("Validation RMSE:", rmse)

# 7. Predict on Kaggle test.csv and save
test_df = pd.read_csv("test.csv")
test_df["NeighborhoodIndex"] = le.transform(test_df["Neighborhood"])

X_test = test_df[["GrLivArea", "BedroomAbvGr", "FullBath", "TotRmsAbvGrd", "NeighborhoodIndex"]]
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Saved predictions to submission.csv")

# 8. Predict for one example house (hard-coded)
# Example details:
# - size: 1800 sq ft
# - 3 bedrooms
# - 2 full bathrooms
# - 7 total rooms
# - neighborhood: CollgCr
example_neighborhood = "CollgCr"
example_neighborhood_index = le.transform([example_neighborhood])[0]

example_data = [[1700, 8, 4, 7, example_neighborhood_index]]
example_price = model.predict(example_data)
print("Predicted price for example house:", example_price[0])
