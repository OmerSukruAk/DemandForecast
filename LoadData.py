import pandas as pd
import numpy as np


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

def read_data():
    sales_data = pd.read_csv('data/sales_data.csv')
    campaign_data = pd.read_csv('data/campaign_data.csv')
    weather_data = pd.read_csv('data/weather_data.csv')
    economic_data = pd.read_csv('data/economic_data.csv')

    merged_data = merge_data(sales_data, campaign_data, weather_data, economic_data)

    all_current_stock_data = pd.read_csv('data/current_stock_data.csv')

    return merged_data, all_current_stock_data