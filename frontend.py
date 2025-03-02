import streamlit as st
from load_model import ProductDemandPrediction
from create_report import image_to_pdf


if st.session_state.get('model') is None:
    st.session_state["model"] = ProductDemandPrediction()

if "ending_stocks" not in st.session_state:
    st.session_state["ending_stocks"] = None

if "enough_stocks" not in st.session_state:        
    st.session_state["enough_stocks"] = None

st.set_page_config(page_title='Product Demand Prediction', page_icon='📦', layout='wide', initial_sidebar_state='auto') # type: ignore

st.markdown("<h1 style='text-align: center;'>Welcome to product demand prediction website</h1>", unsafe_allow_html=True)

col1, space , col2,  = st.columns([2,1,7])

col1.write('This website will help you predict the demand for your product and optimize your stock level')

product_id = col1.selectbox('Please select the product id', range(1, 101))
forecast_days = col1.slider('Please select the number of days you want to forecast', 1, 30, 7)


predict_button = col1.button('Predict')


if predict_button:  
    curr_plot, stocks, prediction_for_next_n_days = st.session_state["model"].predict_demand_and_optimize_stock(forecast_days, product_id)
    col2.pyplot(curr_plot) # type: ignore

    deplation_explanation = st.session_state["model"].get_stock_ending_day(stocks[0],stocks[1:],prediction_for_next_n_days)
    col1.write("Explanations:")
    col1.write(deplation_explanation)




def create_reports():
    ending_stocks = image_to_pdf("AnalysisCharts/stock_is_not_enough/", "reports/Ending_Stocks.pdf")
    enough_stocks = image_to_pdf("AnalysisCharts/stock_is_enough/", "reports/Enough_Stocks.pdf")
    return ending_stocks, enough_stocks

all_report_button = col1.button('Generate reports for all products')

if all_report_button:
    for i in range(1,101):
        curr_plot, stocks, prediction_for_next_n_days = st.session_state["model"].predict_demand_and_optimize_stock(forecast_days, i)
    st.session_state["ending_stocks"], st.session_state["enough_stocks"] = create_reports()


col1_1, col1_2 = col1.columns([1,1])
plchldr1 = col1_1.empty()
plchldr2 = col1_2.empty()

if st.session_state["ending_stocks"] is not None:
    filename = st.session_state["ending_stocks"]
    plchldr1.download_button(
        label="Download Ending_Stocks.pdf",
        data=open(f"reports/{filename}", "rb").read(),
        file_name = st.session_state["ending_stocks"],
        mime="application/pdf"
    )

if st.session_state["enough_stocks"] is not None:
    filename = st.session_state["enough_stocks"]
    plchldr2.download_button(
        label="Download Enough_Stocks.pdf",
        data=open(f"reports/{filename}", "rb").read(),
        file_name = st.session_state["enough_stocks"],
        mime="application/pdf"
    )