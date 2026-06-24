import streamlit as st
import joblib
model=joblib.load('early_warning_business_failure_with_objects.pkl')
l1=joblib.load('le1.pkl')
l2=joblib.load('le2.pkl')
l3=joblib.load('le3.pkl')
l4=joblib.load('le4.pkl')
s=joblib.load('sc.pkl')

st.title('Early Warning Business Failure Prediction')
st.write('Enter Data Description')

Company_age=st.number_input('Enter Company Age')
Revenue_growth=st.number_input('Enter revenue growth')
profit_margin=st.number_input('Enter profit margin')
debt_to_equity=st.number_input('Enter debt to equity')
current_ratio=st.number_input('Enter current ratio')
cash_flow=st.number_input('Enter cashflow')
interest_coverage=st.number_input('Enter interest coverage')
employee_count=st.number_input('Enter employee count')
market_volatility=st.number_input('Enter market volatality')
payment_delays=st.number_input('Enter payment delays')
industry=st.selectbox('Select industry',['Retail','Construction','Logistics','Healthcare','Manufacturing','IT Services'])
company_size=st.selectbox('Select company size',['Small','Medium','Large'])
region=st.selectbox('Select region',['North','East','West','South'])
ownership_type=st.selectbox('Select Ownership Type',['Public','Private','Partnership','Startup'])

industry=l1.transform([industry])[0]
company_size=l2.transform([company_size])[0]
region=l3.transform([region])[0]
ownership_type=l4.transform([ownership_type])[0]
if st.button('predict'):
    result=model.predict(s.transform([[Company_age,Revenue_growth,profit_margin,debt_to_equity,current_ratio,cash_flow,interest_coverage,employee_count,market_volatility,payment_delays,industry,company_size,region,ownership_type]]))[0]
    st.success('the output is {}'.format(result))
    


