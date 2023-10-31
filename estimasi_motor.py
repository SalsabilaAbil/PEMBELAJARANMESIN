import pickle
import streamlit as st

model = pickle.load(open('estimasi_motor.sav', 'rb'))

st.title('Estimasi Keuntungan Penjualan Motor Di Eropa')

Year = st.number_input('Input Tahun Motor')
Unit_Cost = st.number_input('Input Biaya Unit')
Order_Quantity = st.number_input('Input Jumlah Pesanan')
Day = st.number_input('Input Day')

prediksi = ''
if st.button('Estimasi Jumlah Keuntungan'):
    prediksi = model.predict(
        [[Year, Unit_Cost, Order_Quantity, Day]]
    )
    st.write('Estimasi Keuntungan Penjualan Motor di Eropa : ', prediksi)
