# Bitcoin-price
Bit coin price prediction with regression
Bitcoin Price Prediction Project - Explanation
This project aims to analyze and predict Bitcoin prices using historical data. It involves data preprocessing, exploratory analysis, and machine learning (Linear Regression) to predict future Bitcoin prices.
 Step 1: Load and Preprocess Data
What i did:
Loaded the dataset and checked its structure.
Converted the "date" column to datetime format for time series analysis.
Set "date" as the index to make it easier to visualize trends.
Selected independent (X) and dependent (y) variables for model training:
Features (X): "open", "high", "low"
Target (y): "close" (we want to predict this)
Step 2: Data Visualization & Analysis
What I analyzed:
Bitcoin Closing Price Over Time → Showed how Bitcoin's price fluctuated.
High & Low Prices Over Time → Compared Bitcoin's highest and lowest prices daily.
Trading Volume Over Time → Checked how much Bitcoin was traded.
Correlation Heatmap → Showed how different features (open, high, low, close prices, volume) are related.
Key Insights from Visualizations:
Bitcoin prices follow an up-and-down trend, with volatility.
High and low prices are closely related to closing prices, meaning they strongly impact Bitcoin’s final value.
Trading volume sometimes spikes during major price changes.
Step 3: Train a Linear Regression Model
Why Linear Regression?
Linear Regression predicts the future price (Close price) using existing features (Open, High, Low).
Training the Model
python code
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
Step 4: Interpret the Model's Predictions
If MAE and MSE are low, the model is making accurate predictions.
If errors are high, we might need to improve the model using:
More advanced models like LSTMs (Neural Networks)
More features (e.g., trading volume, moving averages)
Summary
 Goal: Predict Bitcoin prices using historical data.
Steps Taken:

Loaded & cleaned the data 
Visualized trends 
Built a prediction model 
 Key Learnings:
Bitcoin prices are highly volatile.
Linear Regression can predict prices based on past data.
More complex models may be needed for better accuracy.
