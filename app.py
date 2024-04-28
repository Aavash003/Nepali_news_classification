import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

st.title("नेपाली समाचार वर्गीकरण ")
df = pd.read_csv("Cleaned_nepali_news_datasets .csv",usecols=["सफा समाचार पाठ","श्रेणी"])
df

st.write("\n")
st.write("\n")
st.write("डाटा भिजुअलाइजेशन")

st.bar_chart(df["श्रेणी"].value_counts())

st.write("\n")
st.write("\n")
st.line_chart(df["श्रेणी"].value_counts())



st.header("समाचार प्रविष्ट गर्नुहोस् : ")

# Training model
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer()
X = df['सफा समाचार पाठ']
y = df['श्रेणी']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05) #Splitting dataset

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=700)),
                     ('clf', LogisticRegression(random_state=1,max_iter=1000))])

#training the model
model = pipeline.fit(X_train,y_train)


#taking the data from the user
file = open('news.txt','r',encoding="utf-8")
news = file.read()
file.close()

#news = st.read("Enter news = ")
news = st.text_area("समाचार प्रविष्ट गर्नुहोस् : ")

if  st.button("पेश गर्नुहोस्"):


	if news != "":
		#news = input("Enter news = ")
		news_data = {'predict_news':[news]}
		news_data_df = pd.DataFrame(news_data)

		predict_news_cat = model.predict(news_data_df['predict_news'])

		# st.write(predict_news_cat)
		st.write("अनुमानित समाचार श्रेणी = ",predict_news_cat[0])

	else:
		st.write("फाइलमा कुनै समाचार वा डाटा छैन!!")