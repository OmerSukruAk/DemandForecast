import pandas as pd
import numpy as np
from LoadData import read_data
import joblib 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from config import window_size


def sliding_window(data, window_size, product_id):
    result = []
    data = data[data['product_id'] == product_id]
    #print(len(data))
    if len(data) > 0:  # Check if product_id exists in the DataFrame
        for i in range(len(data) - window_size + 1):
            window = {'date': data['date'].iloc[i + window_size - 1]}
            for j in range(window_size-1):
                window[f"sales_quantity_{j+1}_day_before"] = data['sales_quantity'].iloc[i+j]
            result.append(window)

        result = pd.DataFrame(result)
        result = result.merge(data, on='date', how='left')
    else:
        print(f"No data found for product_id: {product_id}")
        result = pd.DataFrame()  # Return an empty DataFrame if product_id does not exist

    return result

def demand_forecast_and_stock_optimization(merged_df, window_size, product_id):

    n_windowed_df = sliding_window(merged_df, window_size=window_size, product_id=product_id)
    n_windowed_df = n_windowed_df[n_windowed_df['product_id'] == product_id]
    n_windowed_df.drop_duplicates(inplace=True)
    n_windowed_df['average_temperature'] = n_windowed_df['average_temperature'].astype(int)
    n_windowed_df['inflation_rate'] = n_windowed_df['inflation_rate'].astype(int)
    n_windowed_df['unemployment_rate'] = n_windowed_df['unemployment_rate'].astype(int)

    X = n_windowed_df.drop(columns=['product_id', 'date', 'sales_quantity'])
    y = n_windowed_df['sales_quantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "SVR": SVR()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(mae, mse, rmse)

        results[name] = mae

    if results:
        best_model_name = min(results, key=results.get) # type: ignore
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        demand = y_pred.mean()
        ordering_cost = 50
        holding_cost = 10
        eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost)

    else:
        print("No models were trained successfully.")
    return best_model, eoq, results # type: ignore


# Load Data

merged_data, all_current_stock_data = read_data()
window_size_val = window_size

for product_id in range(1,101): 

    demand_model, optimal_stock,sales_data = demand_forecast_and_stock_optimization(merged_data, window_size_val, product_id)

    save_model_name = f"models/demand_model_id_w{window_size_val}_{product_id}.pkl"
    joblib.dump(demand_model, save_model_name)

    print(f"Model for product_id: {product_id} saved as {save_model_name}")