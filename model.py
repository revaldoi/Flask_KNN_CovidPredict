# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

covid = pd.read_csv('covid_dataset.csv')

covid = covid.drop(['id', 'patient_type', 'entry_date', 'date_symptoms', 'date_died', 'other_disease', 'icu', 'inmsupr'], axis=1)

df = covid[~covid[['sex','age','intubed','pneumonia','pregnancy','diabetes','copd','asthma','hypertension','cardiovascular','obesity','renal_chronic','tobacco','contact_other_covid','covid_res']].isin(['97','98','99','3'])].dropna()

pd.options.display.float_format = '{:,.0f}'.format
df = df.astype(int)

atribut = df[['sex','age','intubed','pneumonia','pregnancy','diabetes','copd','asthma','hypertension','cardiovascular','obesity','renal_chronic','tobacco','contact_other_covid']]
label = df['covid_res']

from sklearn.neighbors import KNeighborsClassifier
regressor = KNeighborsClassifier()

regressor.fit(atribut,label)

print(regressor.score(atribut,label))

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[2,21,2,2,2,2,2,2,2,2,2,2,2,1]]))
