import streamlit as st
from load_model import ProductDemandPrediction
from config import window_size,next_n_days


if st.session_state.get('model') is None:
    st.session_state["model"] = ProductDemandPrediction()



def predict_demand_and_optimize_stock(next_n_days, product_id):

    prediction_for_next_n_days = st.session_state["model"].predict_next_n_days(next_n_day=next_n_days, product_id=product_id)
    print(prediction_for_next_n_days)

    current_stock = st.session_state["model"].get_product_stock_by_id(product_id)
    print(current_stock)

    isEnough = st.session_state["model"].check_if_stock_is_enough(current_stock, prediction_for_next_n_days)
    
    plot, decreasing_stock = st.session_state["model"].plot_stock_depletion(prediction_for_next_n_days, current_stock, product_id,isEnough)

    return plot, decreasing_stock


def get_stock_ending_day (stock, demand):
    day_count = len(demand)
    for i in range(day_count):
        stock = stock - demand[i]
        if stock <= 0:
            return i
    return day_count


st.set_page_config(page_title='Product Demand Prediction', page_icon='📦', layout='wide', initial_sidebar_state='auto') # type: ignore

st.markdown("<h1 style='text-align: center;'>Welcome to product demand prediction website</h1>", unsafe_allow_html=True)

col1, space , col2,  = st.columns([3,1,6])

col1.write('This website will help you predict the demand for your product and optimize your stock level')

product_id = col1.selectbox('Please select the product id', range(1, 101))
forecast_days = col1.slider('Please select the number of days you want to forecast', 1, 30, 7)

predict_button = col1.button('Predict')


if predict_button:  
    curr_plot, stocks = predict_demand_and_optimize_stock(forecast_days, product_id)
    col2.pyplot(curr_plot) # type: ignore

    ending_day = get_stock_ending_day(stocks[0], stocks[1:])
    col2.write(f"Stock will be depleted in day {ending_day}")