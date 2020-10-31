import numpy as np
import pandas as pd
import streamlit as st

import pickle
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)
pickle_in.close()

pickle_in=open('scale.pkl','rb')
sc=pickle.load(pickle_in)
pickle_in.close()



def predict_survival(Pclass,Sex,Age,Total_fare,Family_size,Queenstown,Southampton):
    x=np.array([[Pclass,Sex,Age,np.log(Total_fare),Family_size,Queenstown,Southampton]],dtype=float)
    x=(x-sc.mean_)/sc.scale_
    if classifier.predict(x).astype(int):
        return 'congratulations you have Survived'
    else:
        return 'The world will mourn for you'


def main():
    html_temp = '''
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Titanic Survival Prediction</h2>
    </div>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    st.subheader("Lets's see whether you would have survived in titanic or not?")
    Pclass=st.text_input('Passenger Class (upper-1,Middle-2,Lower-3)')
    Sex=st.radio('Sex',['Male','Female'])
    if Sex=='Male':
        Sex=0
    else:
        Sex=1
    Age=st.number_input('Enter your Age',min_value=0,value=0)
    Family_size=st.number_input('No. of family members onboarded titanic',min_value=0,value=0)
    Total_fare=st.number_input('Total Fare',min_value=0.0)
    Embarked=st.radio('Embarked',['Queenstown','Southampton','Cherbourg'])
    if Embarked=='Queenstown':
        Queenstown=1
        Southampton=0
    elif Embarked=='Southampton':
        Queenstown=0
        Southampton=1
    else:
        Queenstown=0
        Southampton=0
    if st.button('Predict'):
        st.write(predict_survival(Pclass,Sex,Age,Total_fare,Family_size,Queenstown,Southampton))
    
    
if __name__=='__main__':
    main()
    
