import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def merge_data(sales_data, campaign_data, weather_data, economic_data):
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    campaign_data['start_date'] = pd.to_datetime(campaign_data['start_date'])
    campaign_data['end_date'] = pd.to_datetime(campaign_data['end_date'])
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    economic_data['date'] = pd.to_datetime(economic_data['date'])

    sales_data = sales_data.merge(weather_data, on='date', how='left')
    sales_data = sales_data.merge(economic_data, on='date', how='left')
    sales_data = sales_data.merge(campaign_data, on='product_id', how='left')

    sales_data['is_campaign'] = ((sales_data['date'] >= sales_data['start_date']) & (sales_data['date'] <= sales_data['end_date'])).astype(int)
    sales_data['discount_rate'] = np.where(sales_data['is_campaign'] == 0, 0, sales_data['discount_rate'])
    sales_data.drop(columns=['start_date', 'end_date', 'campaign_type'], inplace=True)

    sales_data = pd.get_dummies(sales_data, columns=['precipitation'], prefix='precipitation',dtype=int)
    

    sales_data.fillna(method='ffill', inplace=True)

    return sales_data

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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

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
        results[name] = mae
        #print(f"{name} Mean Absolute Error: {mae}")

    if results:
        best_model_name = min(results, key=results.get)
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        demand = y_pred.mean()
        ordering_cost = 50
        holding_cost = 10
        eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost)
        #print(f"Best Model: {best_model_name}")
        #print(f"Economic Order Quantity (EOQ): {eoq}")
    else:
        print("No models were trained successfully.")

    return best_model, eoq, results
def get_product_stock_by_id(all_stock, product_id):
    current_stock_data = all_stock[all_stock['product_id'] == product_id]
    current_stock = current_stock_data['current_stock'].iloc[0]
    return current_stock


def predict_next_n_days(next_n_day, demand_model_pkl):
    temp_array = []
    day_sales_predictions = []

    last_n_days_sales = merged_data[merged_data['product_id'] == product_id].tail(window_size)

    last_day_avg_temp = last_n_days_sales['average_temperature'].iloc[-1]
    last_day_inflation_rate = int(last_n_days_sales['inflation_rate'].iloc[-1])
    last_day_unemployment_rate = int(last_n_days_sales['unemployment_rate'].iloc[-1])
    last_day_campaign = int(last_n_days_sales['is_campaign'].iloc[-1])
    last_day_discount_rate = int(last_n_days_sales['discount_rate'].iloc[-1])
    last_day_is_cloudy = int(last_n_days_sales['precipitation_Cloudy'].iloc[-1])
    last_day_is_rainy = int(last_n_days_sales['precipitation_Rainy'].iloc[-1])
    last_day_is_sunny = int(last_n_days_sales['precipitation_Sunny'].iloc[-1])

    for x in range (0,window_size - 1):
        day_sales_predictions.append(last_n_days_sales['sales_quantity'].iloc[x])

    temp_array.append(last_day_avg_temp)
    temp_array.append(last_day_inflation_rate)
    temp_array.append(last_day_unemployment_rate)
    temp_array.append(last_day_discount_rate)
    temp_array.append(last_day_campaign)
    temp_array.append(last_day_is_cloudy)
    temp_array.append(last_day_is_rainy)
    temp_array.append(last_day_is_sunny)

    predictions = []

    for _ in range(next_n_day):
        predict_data = [day_sales_predictions + temp_array]
        #print(predict_data)
        prediction = int(demand_model_pkl.predict(predict_data)[0])
        #print(prediction)
        predictions.append(prediction)
        day_sales_predictions.pop(0)
        day_sales_predictions.append(prediction)
        #print(day_sales_predictions)
    
    return predictions
    
def check_if_stock_is_enough(stock, demand, optimal_stock):
    if stock >= demand:
        return "Stock is enough"
    elif stock < demand and stock >= optimal_stock:
        return "Stock is not enough. But it is more than Economic Order Quantity. Please order more products"
    else:
        return "Stock is not enough. Please order more products"


# Verileri yükleme
sales_data = pd.read_csv('data/sales_data.csv')
campaign_data = pd.read_csv('data/campaign_data.csv')
weather_data = pd.read_csv('data/weather_data.csv')
economic_data = pd.read_csv('data/economic_data.csv')

all_current_stock_data = pd.read_csv('data/current_stock_data.csv')


merged_data = merge_data(sales_data, campaign_data, weather_data, economic_data)


# Modeli çalıştırma
product_id = 1
merged_data = merge_data(sales_data, campaign_data, weather_data, economic_data)

# demand_model_resp, optimal_stock_resp,sales_data_resp = demand_forecast_and_stock_optimization(merged_data, window_size=3, product_id=3)

window_size = 8
print(f"Window Size: {window_size}")
demand_model, optimal_stock,sales_data = demand_forecast_and_stock_optimization(merged_data, window_size, product_id)
print("\n")

import joblib 

joblib.dump(demand_model, 'demand_model.pkl')

demand_model_pkl = joblib.load('demand_model.pkl') 

last_n_days_sales = merged_data[merged_data['product_id'] == product_id].tail(window_size)
#print(last_n_days_sales)

current_stock = get_product_stock_by_id(all_current_stock_data, product_id)

prediction_for_next_n_days = predict_next_n_days(next_n_day=15, demand_model_pkl=demand_model_pkl)

print(check_if_stock_is_enough(current_stock, sum(prediction_for_next_n_days), 150))