import streamlit as st
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import numpy as np
st.write("""
# Stock Market Web Application **Visually** show data on a stock! Data range from Jan 1, 20201- Dec, 2022
""")
st.title('Stock price prediction')
image=Image.open("C:/Users/P Balaji/OneDrive/Desktop/Final Project/DVD/WebApplication/WhatsApp Image 2023-04-29 at 16.15.26.jpg")
st.image(image, use_column_width=True)
st.sidebar.header('User Input')
def get_input():
  start_date=st.sidebar.text_input("Start Date","10-01-2022")
  end_date=st.sidebar.text_input("End Date"," 09-01-2023")
  stock_symbol=st.sidebar.text_input("Stock symbol","RELIANCE")
  return start_date,end_date,stock_symbol
def get_company_name(symbol):
  if symbol=="RELIANCE":
    return 'RELIANCE'
  elif symbol=='TSLA':
    return 'Tesla'
  elif symbol=='GOOG':
    return 'Alphabet'
  else:
    'None'

def get_data(symbol,start,end):
  if symbol.upper()=='RELIANCE':
    df=pd.read_csv("C:/Users/P Balaji/OneDrive/Desktop/Final Project/DVD/WebApplication/RELIANCENS.csv")
  elif symbol.upper()=='GOOG':
    df=pd.read_csv("C:/Users/P Balaji/OneDrive/Desktop/Final Project/DVD/WebApplication/GOOG.csv")
  elif symbol.upper()=='TSLA':
    df=pd.read_csv("C:/Users/P Balaji/OneDrive/Desktop/Final Project/DVD/WebApplication/TSLA (1).csv")
  else:
    df=pd.DataFrame(columns=['Date','Close','Open','Volume','Adjclose','High','Low'])
 
  start=pd.to_datetime(start)
  end=pd.to_datetime(end)

  start_row=0
  end_row=0
  for i in range(0,len(df)):
    if start<=pd.to_datetime(df['Date'][i] ):
      start_row=i
      break
  for j in range(0,len(df)):
    if end>=pd.to_datetime(df['Date'][len(df)-1-j]):
      end_row=len(df)-1-j
      break

  df=df.set_index(pd.DatetimeIndex(df['Date'].values))
  return df.iloc[start_row:end_row +1, :]

start, end, symbol=get_input()
df=get_data(symbol,start,end)
company_name=get_company_name(symbol.upper())
st.header(company_name+" Open price\n")
st.line_chart(df['Open'])


st.header(company_name+" Volume\n")
st.line_chart(df['Volume'])

st.header('Data Statistics')
st.write(df.describe())

df = df.drop(columns=['Date'])

scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)   

y_open = df[:,1]
y_open
#splitting data set into 80 for testing
ntrain = int(len(y_open)*0.8) 

train = df[0:ntrain]
test  = df[ntrain:len(df)]
print(train)
y_open_train = y_open[0:ntrain]
y_open_test  = y_open[ntrain:len(y_open)]
def to_sequences_1(seq_size, data,open):
    x = []
    y = []
    #we have mapped for 10 days duration to increase the accuracy of model
    #10 sets of x class and y sets of y class were mapped as a single unit object
    #instead of training model over single instace i.e for single day data
    #we have trained for 10 different instance treating them as single unit
    for i in range(len(data)-seq_size-1):
        window = data[i:(i+seq_size)]
        after_window = open[i+seq_size]
        window = [[x] for x in window]
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)


timesteps = 10

x_train, y_train = to_sequences_1(timesteps, train, y_open_train)
x_test, y_test   = to_sequences_1(timesteps, test, y_open_test)

x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[2], x_train.shape[1],x_train.shape[3]))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[2],x_test.shape[1],x_test.shape[3]))

model=load_model("C:/Users/P Balaji/OneDrive/Desktop/Final Project/DVD/WebApplication/CNN_Open.hdf5")
pred = model.predict(x_test)
st.header('Predictions vs Actual')

fig2=plt.figure(figsize=(12,6))
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.plot(y_test, label = 'actual')
plt.plot(pred,   label = 'predicted')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot()