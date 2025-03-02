import joblib
from LoadData import read_data
from config import window_size,next_n_days
import matplotlib.pyplot as plt

class ProductDemandPrediction:

    def __init__(self):
        self.window_size = window_size
        self.next_n_days = next_n_days
        self.all_models  = self.load_all_models()
        self.merged_data, self.all_current_stock_data = read_data()
    
    def load_all_models(self):
        all_models = {}
        for product_id in range(1,101):
            all_models[product_id] = joblib.load(f'models/demand_model_id_w{self.window_size}_{product_id}.pkl')
        return all_models

    def get_product_stock_by_id(self, product_id):
        current_stock_data = self.all_current_stock_data[self.all_current_stock_data['product_id'] == product_id]
        current_stock = current_stock_data['current_stock'].iloc[0]
        return current_stock

    def predict_next_n_days(self, next_n_day, product_id):
        temp_array = []
        day_sales_predictions = []

        last_n_days_sales = self.merged_data[self.merged_data['product_id'] == product_id].tail(self.window_size)

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
            prediction = int(self.all_models[product_id].predict(predict_data)[0])
            predictions.append(prediction)
            day_sales_predictions.pop(0)
            day_sales_predictions.append(prediction)
        
        return predictions
        
    def check_if_stock_is_enough(self,stock, demand):
        # day_count = len(demand)
        # response = f"Current Stock Level: {stock}\n Demand Predictions for the next {day_count} days: {demand} \n Decision: " 
        if stock >= sum(demand):
            return True
            #response += f"Stock is enough for the next {day_count} days."
        else:
            #response += f"Stock is not enough. Please order more products. For the next {day_count} days, you need {sum(demand) - stock} more products to meet the demand minimum"        
            return False


    def plot_stock_depletion(self, demand, initial_stock,product_id, isStockEnough):
        
        filename = ''
        if isStockEnough:
            filename = f'AnalysisCharts/stock_is_enough/{product_id}id_next{next_n_days}days.png'
        else:
            filename = f'AnalysisCharts/stock_is_not_enough/{product_id}id_next{next_n_days}days.png'

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
        

        return plt, stock


model = ProductDemandPrediction()

for product_id in range(1,101): 

    prediction_for_next_n_days = model.predict_next_n_days(next_n_day=next_n_days, product_id=product_id)
    print(prediction_for_next_n_days)

    current_stock = model.get_product_stock_by_id(product_id)
    print(current_stock)

    isEnough = model.check_if_stock_is_enough(current_stock, prediction_for_next_n_days)
    
    model.plot_stock_depletion(prediction_for_next_n_days, current_stock, product_id,isEnough)
