import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_dataset():

    products = pd.DataFrame({
        'product_id': range(1, 101),
        'category': random.choices(['Clothing', 'Accessory', 'Shoes'], k=100),
        'sub_category': [random.choice(['T-shirt', 'Pants', 'Dress', 'Hat', 'Bag', 'Sports Shoes', 'High Heels']) for _ in range(100)],
        'color': [random.choice(['Red', 'Blue', 'Black', 'White']) for _ in range(100)],
        'size': [random.choice(['S', 'M', 'L', 'XL']) for _ in range(100)]
    })

    # Date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 1, 2)
    date_range = pd.date_range(start_date, end_date, freq='D')
    

    # Sales data
    daily_sales_data = []
    for product_id in products['product_id']:
        for date in date_range:
            sales_quantity = max(0, int(np.random.normal(loc=50, scale=20)))
            daily_sales_data.append({
                'product_id': product_id,
                'date': date,
                'sales_quantity': sales_quantity
            })
    daily_sales_data = pd.DataFrame(daily_sales_data)
    
    monthly_sales_data = daily_sales_data.groupby([pd.Grouper(key='date', freq='M'), 'product_id']).sum().reset_index()

    current_stock_data = []
    previous_stock = 400  # Initial stock value

    consecutive_zero_days = 0  # Counter for consecutive zero stock days
    for product_id in products['product_id']:
        for date in date_range:
            sales = int(np.random.normal(loc=20, scale=5))  # Example sales data
            current_stock = max(0, previous_stock - sales)
            
            if current_stock == 0:
                consecutive_zero_days += 1
            else:
                consecutive_zero_days = 0
            
            if consecutive_zero_days == 5:
                current_stock = 300  # Initialize current stock as 100
                
            current_stock_data.append({
                'product_id': product_id,
                'date': date,
                'current_stock': current_stock
            })
            
            previous_stock = current_stock  # Update previous stock for the next day
    current_stock_data = pd.DataFrame(current_stock_data)

    # Campaign data
    campaign_data = []
    for _ in range(150):
        product_id = random.choice(products['product_id'])
        start_date_campaign = random.choice(date_range)
        end_date_campaign = start_date_campaign + timedelta(days=random.randint(1, 5))
        discount_rate = random.choice([10, 20, 30, 50])
        campaign_type = random.choice(['% Discount in Cart', 'Buy 2 Get 1 Free'])
        if campaign_type == 'Buy 2 Get 1 Free':
            discount_rate = 33
        campaign_data.append({
            'product_id': product_id,
            'start_date': start_date_campaign,
            'end_date': end_date_campaign,
            'discount_rate': discount_rate,
            'campaign_type': campaign_type
        })
    campaign_data = pd.DataFrame(campaign_data)

    # Weather data (optional)
    weather_data = []
    for date in date_range:
        average_temperature = np.random.uniform(5, 30)
        precipitation = random.choice(['Rainy', 'Cloudy', 'Sunny'])
        weather_data.append({
            'date': date,
            'average_temperature': average_temperature,
            'precipitation': precipitation
        })
    weather_data = pd.DataFrame(weather_data)

    # Economic data
    economic_data = []
    months = pd.date_range(start_date, end_date, freq='MS')
    for month in months:
        inflation_rate = np.random.uniform(0, 20)
        unemployment_rate = np.random.uniform(3, 15)
        economic_data.append({
            'date': month,
            'inflation_rate': inflation_rate,
            'unemployment_rate': unemployment_rate,
        })
    economic_data = pd.DataFrame(economic_data)

    return products, daily_sales_data,monthly_sales_data, current_stock_data, campaign_data, weather_data, economic_data

# Example usage
products, daily_sales_data,monthly_sales_data, current_stock_data, campaign_data, weather_data, economic_data = create_dataset()

print("Products:", products.head())
print("\nSales Data:", daily_sales_data.head())
print("\nCampaign Data:", campaign_data.head())
print("\nWeather Data:", weather_data.head())
print("\nEconomic Data:", economic_data.head())

daily_sales_data.to_csv('data/sales_data.csv', index=False)
monthly_sales_data.to_csv('data/monthly_sales_data.csv', index=False)
current_stock_data.to_csv('data/current_stock_data.csv', index=False)
campaign_data.to_csv('data/campaign_data.csv', index=False)
weather_data.to_csv('data/weather_data.csv', index=False)
economic_data.to_csv('data/economic_data.csv', index=False)
products.to_csv('data/products.csv', index=False)
