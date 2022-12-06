import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import shap
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
#应用标题
st.set_page_config(page_title='Pred hypertension in  patients with OSA')
st.title('Application and Interpretation Study of Machine Learning Models in Predicting Severe Obstructive Sleep Apnea of Adults')
st.sidebar.markdown('## Variables')
ESSL = st.sidebar.selectbox('Epworth sleepiness scale',('Normal','Low','Middle','High'),index=1)
hypertension = st.sidebar.selectbox('Hypertension',('No','Yes'),index=1)
BQL = st.sidebar.selectbox('Berlin Questionnaire',('Low risk','High risk'),index=1)
SBSL = st.sidebar.selectbox('STOP-Bang questionnaire',('Low risk','High risk'),index=0)
drink = st.sidebar.selectbox('Drinking',('No','Yes'),index=1)
smork = st.sidebar.selectbox('Smorking',('No','Yes'),index=1)
snoring = st.sidebar.selectbox('Course of snoring',('No','Yes'),index=1)
suffocate = st.sidebar.selectbox('Course of choking',('No','Yes'),index=1)
memory = st.sidebar.selectbox('Memory decline',('No','Yes'),index=1)
LOE = st.sidebar.selectbox('Sedentariness',('No','Yes'),index=1)
gender = st.sidebar.selectbox('Gender',('female','male'),index=1)
age = st.sidebar.slider("Age(year)", 0, 99, value=45, step=1)
BMI = st.sidebar.slider("Body mass index", 15.0, 40.0, value=20.0, step=0.1)
waistline = st.sidebar.slider("Waist circumference(cm)", 50.0, 150.0, value=100.0, step=1.0)
NC = st.sidebar.slider("Neck circumference(cm)", 20.0, 60.0, value=30.0, step=0.1)
#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'No':0,'Yes':1,'Normal':0 ,'Low':1, 'Middle':2,'High':3, 'Low risk':0,'High risk':1,'female':0, 'male':1}
ESSL =map[ESSL]
hypertension = map[hypertension]
BQL = map[BQL]
SBSL =map[SBSL]
drink =map[drink]
smork = map[smork]
snoring = map[snoring]
suffocate = map[suffocate]
memory = map[memory]
LOE =map[LOE]
gender = map[gender]
# 数据读取，特征标注
hp_train = pd.read_csv('serve_osa_2.csv')
hp_train['OSAL'] = hp_train['OSAL'].apply(lambda x : +1 if x==1 else 0)
features =["ESSL","hypertension","BQL","SBSL","drink",'smork',"snoring",'suffocate','memory','LOE','gender','age','BMI','waistline','NC']
target = 'OSAL'
random_state_new = 50
X_ros = np.array(hp_train[features])
y_ros = np.array(hp_train[target])
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
gbm.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (gbm.predict_proba(np.array([[ESSL,hypertension,BQL,SBSL,drink,smork,snoring,suffocate,memory,LOE,gender,age,BMI,waistline,NC]]))[0][1])> sp
prob = (gbm.predict_proba(np.array([[ESSL,hypertension,BQL,SBSL,drink,smork,snoring,suffocate,memory,LOE,gender,age,BMI,waistline,NC]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for OSAL:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of High risk group:  '+str(prob)+'%')
    #%%SHAP
    col_names = features
    X_last = pd.DataFrame(np.array([[ESSL,hypertension,BQL,SBSL,drink,smork,snoring,suffocate,memory,LOE,gender,age,BMI,waistline,NC]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    model = gbm
    #%%
    sns.set()
    explainer = shap.Explainer(model, X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot')
    shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
    # sns.set()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                shap_values[a, :], 
                X.iloc[a, :], 
                figsize=(25, 3),
                link = "logit",
                matplotlib=True,
                out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)





