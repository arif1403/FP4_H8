import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.cluster import KMeans

model = pickle.load(open('model/classifier_model.pkl','rb'))
img = Image.open('icon-card.png')
st.set_page_config(page_title="Credit Card User Segmentation",page_icon=img ,layout="centered",initial_sidebar_state="expanded")

html_temp = """ 
    <div style ="background-color:darkblue;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Credit Card User Segmentation</h1> 
    </div> 
    """

# display the front end aspect
st.markdown(html_temp, unsafe_allow_html=True)

# st.title('Credit Card User Segmentation')
st.write("""
----
""")

def user_input_parameters():
    # col1, col2 = st.columns([3])
    # with col1:
    balance = st.number_input('Balance :')
    balance_frq = st.number_input('Balance Frequency :')
    purchases = st.number_input('Purchases Amount :')
    oneoff_purchases = st.number_input('ONEOFF Purchases Amount :')
    installment_purchases = st.number_input('Installments Purchases Amount :')
    cash_advance = st.number_input('Cash Advance Amount :')
    purchases_frq = st.number_input('Purchases Frequency :')
    oneoff_purchases_frq = st.number_input('ONEOFF Purchases Frequency :')
    installment_purchases_frq = st.number_input('Purchases Installments Frequency :')
    # with col2:
    cash_advance_frq = st.number_input('Cash Advance Frequency :')
    cash_advance_trx = st.number_input('Cash Advance Transaction :')
    purchases_trx = st.number_input('Purchases Transaction :')
    credit_limit = st.number_input('Credit Limit Amount :')
    payments = st.number_input('Payments Amount :')
    minimum_payments = st.number_input('Minimum Payments Amount :')
    prc_full_payment = st.number_input('Percentage of Full Payments :')
    tenure = st.number_input('Tenure :')

    data = {
        'balance': balance,
        'balance_frq': balance_frq,
        'purchases': purchases,
        'oneoff_purchases': oneoff_purchases,
        'installment_purchases': installment_purchases,
        'cash_advance': cash_advance,
        'purchases_frq': purchases_frq,
        'oneoff_purchases_frq': oneoff_purchases_frq,
        'installment_purchases_frq': installment_purchases_frq,
        'cash_advance_frq': cash_advance_frq,
        'cash_advance_trx': cash_advance_trx,
        'purchases_trx': purchases_trx,
        'credit_limit': credit_limit,
        'payments': payments,
        'minimum_payments': minimum_payments,
        'prc_full_payment': prc_full_payment,
        'tenure': tenure
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_parameters()

# st.subheader('User Input Parameters')
# st.write(df)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)


if st.button("Predict"):
    if prediction[0] == 0:
        st.subheader('Prediction')
        st.write(prediction)
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        st.success('Cluster 1 Tipe user yang memiliki balance moderat, sangat jarang melakukan transaksi pembelian, lebih sering melakukan transaksi dengan uang tunai dimuka, hampir tidak pernah melakukan pembelian dengan metode mencicil. Tipe user ini memiliki limit kartu kredit medium, tipikal user dengan tingkat periode tenur paling lama.')
    elif prediction[0] == 1:
        st.subheader('Prediction')
        st.write(prediction)
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        st.success('Cluster 2 Memiliki balance dan limit kartu kredit paling tinggi, lebih sering melakukan pembelian dengan metode sekali bayar(one off purchases), sering melakukan transaksi belanja, hampir tidak pernah melakukan pembelian dengan uang tunai dimuka serta tipikal user dengan periode tenure.')
    elif prediction[0] == 2:
        st.subheader('Prediction')
        st.write(prediction)
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        st.success('Cluster 3 Memiliki balance paling rendah diantara cluster lain, frekuensi pembelian cukup tinggi dan sering melakukan pembelian dengan metode pembayaran mencicil, memiliki limit kartu kredit paling rendah, user ini cenderung memiliki periode tenure medium.')
    else:
        st.error('Users are not included in any Cluster!')

st.sidebar.title('Informasi :')
st.sidebar.info('Pada Prediction dan Prediction Probability jika kolom berlabel [0] maka Cluster 1, '
        'kemudian apabila kolom berlabel [1] maka Cluster 2 dan kolom berlabel [2] maka Cluster 3')
st.sidebar.info('Aplikasi ini digunakan untuk menunjukkan segmentasi pelanggan dengan mengimplementasikan analisa clustering.')