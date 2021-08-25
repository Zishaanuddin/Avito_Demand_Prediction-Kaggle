# -*- coding: utf-8 -*-
import streamlit as st

import numpy as np
import pandas as pd
import pickle
#import cv2
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


max_length_title = 8
max_length_desc = 200

def categorical_encoder(data):
    '''This function encode the categorical feature which we will use in NN along with embedding layer'''
    name_list = ["user_type","parent_category_name","category_name","region","city","param_1", "param_2", "param_3","image_top_1"]
    values = []
    for (name, val) in zip(name_list, data):
        tokeniser = pickle.load(open(name+'tokeniser.pkl', 'rb'))
        print(name, val)
        encode = tokeniser.texts_to_sequences(str(val))
        encode = np.array(encode).astype(np.float64)
        values.append(encode)
        
    return values

def feature_engineering(price, title_str, desc_str):
    '''This function creates the new feature with aggregating feature with mean, median, sum, min, and max'''
    
    # making aggregates feature with mean, sum and max.
    price = np.array(price)
    scaler = pickle.load(open('price.pkl', 'rb'))
    price = scaler.transform(price.reshape(1,-1))
    price = price.flatten()
    
    data = {'price': price, 'title_words_length': len(title_str.split()), 
            'description_words_length': len(desc_str.split()),'symbol1_count': desc_str.count('↓'), 
            "symbol2_count": desc_str.count('\*'),'symbol3_count': desc_str.count('✔'), "symbol4_count": desc_str.count('❀'),
            "symbol5_count": desc_str.count('➚'),"symbol6_count":  desc_str.count('ஜ'), "symbol7_count": desc_str.count('.'),
            "symbol8_count":desc_str.count('!'), "symbol9_count": desc_str.count('\?'), "symbol10_count": desc_str.count('  '), 
            "symbol11_count": desc_str.count('-'), "symbol12_count": desc_str.count(',')}  
    
    dataframe = pd.DataFrame(data, index=[0])
    
    return dataframe

def encoder(title, desc):

  ''' This function perform the tokenization and then convert words to integers and then perform padding and returns the values '''
    # integer encode
  tokeniser_title = pickle.load(open('tokeniser_title.pkl', 'rb'))
  tokeniser_desc = pickle.load(open('tokeniser_desc.pkl', 'rb'))
  
  encoded_title = tokeniser_title.texts_to_sequences(title)
  encoded_desc = tokeniser_desc.texts_to_sequences(desc)


  padded_title = np.array(pad_sequences(encoded_title, maxlen=max_length_title, padding='post')).astype(np.float64)
  padded_desc = np.array(pad_sequences(encoded_desc, maxlen=max_length_desc, padding='post')).astype(np.float64)


  return padded_title, padded_desc



def Prediction(title_ru, desc_ru, Price, User, par_cat, cat_name, Region, City, Param_1 , Param_2, Param_3, Image_top_1):
    cat_var = [User, par_cat, cat_name, Region, City, Param_1 , Param_2, Param_3, Image_top_1]
    encoded = encoder(title_ru, desc_ru)
    cat_vars = categorical_encoder(cat_var)
    num_vars = feature_engineering(Price, title_ru, desc_ru)
    
    final_feature = [encoded]
    final_feature = final_feature + cat_vars
    final_feature = final_feature.append(num_vars.to_numpy().astype(np.float64))
    
    path = 'baseline_model_v1.weights.best.hdf5'
    
    model = tf.keras.models.load_model(path)
    pred = model.predict(final_feature)
    
    return pred


def main():
    html_temp = """ 
    <div style ="background-color:tomato;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Avito's Demand Prediction App</h1> 
    </div> 
    """
    # display the front end aspect
    region = pickle.load(open('region.pkl', 'rb'))
    city = pickle.load(open('city.pkl', 'rb'))
    parent_category = pickle.load(open('parent_category_name.pkl', 'rb'))
    category_name = pickle.load(open('category_name.pkl', 'rb'))
    param_1 = pickle.load(open('param_1.pkl', 'rb'))
    param_2 = pickle.load(open('param_2.pkl', 'rb'))
    param_3 = pickle.load(open('param_3.pkl', 'rb'))
    user_type = pickle.load(open('user_type.pkl', 'rb'))
    image_top_1 = pickle.load(open('image_top_1.pkl', 'rb'))
    
    st.markdown(html_temp, unsafe_allow_html = True)
    
    title_ru = st.text_input("Ad Title in Russian")
    desc_ru = st.text_input("Ad Description in Russian")
    User = st.selectbox('Select your User Type', (user_type)) 
    par_cat = st.selectbox('Select your Parent Category', (parent_category))
    cat_name = st.selectbox('Select your Sub-Category', (category_name))  
    Region = st.selectbox('Select your Region', (region))    
    City = st.selectbox('Select your City', (city))
    Price = st.number_input('Select your Price')
    Param_1 = st.selectbox('Select your Parameter 1', (param_1)) 
    Param_2 = st.selectbox('Select your Parameter 2', (param_2)) 
    Param_3 = st.selectbox('Select your Parameter 3', (param_3))
    Image_top_1 = st.selectbox('Select your Image Code', (image_top_1))
        
    
    if st.button("predict"):
        result = Prediction(title_ru, desc_ru, Price, User, par_cat, cat_name, Region, City, Param_1 , Param_2, Param_3, Image_top_1)
        st.success('Your deal probablity is {}'.format(result))

    
if __name__=='__main__':
    main()

    
    
    
    