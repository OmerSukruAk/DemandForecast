import joblib
from LoadData import read_data
from config import window_size,next_n_days


def get_product_stock_by_id(all_stock, product_id):
    current_stock_data = all_stock[all_stock['product_id'] == product_id]
    current_stock = current_stock_data['current_stock'].iloc[0]
    return current_stock

def predict_next_n_days(merged_data, next_n_day, demand_model_pkl,window_size):
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
        prediction = int(demand_model_pkl.predict(predict_data)[0])
        predictions.append(prediction)
        day_sales_predictions.pop(0)
        day_sales_predictions.append(prediction)
    
    return predictions
    
def check_if_stock_is_enough(stock, demand):
    day_count = len(demand)
    response = f"Current Stock Level: {stock}\n Demand Predictions for the next {day_count} days: {demand} \n Decision: " 
    if stock >= sum(demand):
        return True
        #response += f"Stock is enough for the next {day_count} days."
    else:
        #response += f"Stock is not enough. Please order more products. For the next {day_count} days, you need {sum(demand) - stock} more products to meet the demand minimum"        
        return False

import matplotlib.pyplot as plt

def plot_stock_depletion(demand, initial_stock,product_id, filename='stock_depletion.png'):
    stock = [initial_stock]
    for d in demand:
        stock.append(max(stock[-1] - d, 0))  # Ensure stock doesn't go negative
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(stock)), stock, marker='o', linestyle='-', label=f'Stock Level for {product_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Stock')
    plt.title('Stock Depletion Over Time')
    plt.xticks(range(len(stock)))
    plt.grid(True)
    plt.legend()
    plt.savefig(filename) 


window_size_val = window_size
merged_data, all_current_stock_data = read_data()

for product_id in range(1,101): 

    demand_model_pkl = joblib.load(f'models/demand_model_id{product_id}_w{window_size_val}_.pkl') 
    last_n_days_sales = merged_data[merged_data['product_id'] == product_id].tail(window_size_val) # type: ignore
    current_stock = get_product_stock_by_id(all_current_stock_data, product_id)

    prediction_for_next_n_days = predict_next_n_days(merged_data=merged_data,next_n_day=next_n_days, demand_model_pkl=demand_model_pkl, window_size=window_size_val)
    print(prediction_for_next_n_days)
    print(current_stock)

    isEnough = check_if_stock_is_enough(current_stock, prediction_for_next_n_days)
    analysis_chart_filename = ''
    if isEnough:
        analysis_chart_filename = f'AnalysisCharts/stock_is_enough_{product_id}id_next{next_n_days}days.png'
    else:
        analysis_chart_filename = f'AnalysisCharts/stock_is_not_enough_{product_id}id_next{next_n_days}days.png'
    plot_stock_depletion(prediction_for_next_n_days, current_stock, product_id,f'AnalysisCharts/stock_depletion_{product_id}id_next{next_n_days}days.png')

