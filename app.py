import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title ='fake News Detector')

model=pickle.load(open('logistic.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))
explainer=pickle.load(open('shap.pkl','rb'))


sia=SentimentIntensityAnalyzer()

category_keywords={
    'politics':['election','president','government','congress','minister'],
    'sports':['match','tournament','player','championship','league','tennis','football'],
    'business':['market','stock','economy','company','trade','finance'],
    'entertainment':['movie','celebrity','music','film','actor','show'],
    'health':['hospital','disease','vaccine','doctor','health','covid'],
    'world':['country','nation','international','war','conflict']
}

def detct_category(text):
    text_lower=text.lower()
    score={cat:sum(text_lower.count(word) for word in words)
           for cat,words in category_keywords.items()}
    best_cat=max(score,key=score.get)
    return best_cat if score[best_cat] > 0 else 'general'

def get_sentiment(text):
    scores=sia.polarity_scores(text)
    compound=scores['compound']

    if compound >= 0.05:
        label='Positive'
    elif compound <= -0.05:
        label='Negative'
    else:
        label ='Neutral'
    return label,compound

st.title('Fake News Detector')
st.write('Paste a news headline and the model will detect')
with st.sidebar:
    st.info('This is a News detector App')
    st.success('It is build using NLP{TF-IDF} + Logistic Regression')
    st.success('It is build by Atharva ')


user_txt=st.text_area('Enter the news text here')

if st.button('Analyze'):
    if user_txt.strip() =="":
        st.warning('Please enter some text first')
    else:
        X_input=tfidf.transform([user_txt])
        prediction=model.predict(X_input)[0]
        proba=model.predict_proba(X_input)[0]

        label_map={0:'Real',1:'Fake'}
        pred_label=label_map[prediction]
        confidence=proba[prediction] * 100


        col1,col2,col3=st.columns(3)

        with col1:
            if pred_label == 'Fake':
                st.error(f"Prediction:{pred_label}")
            else:
                st.success(f"Prediction:{pred_label}")

        #--------------Sentiment---------------#
        sentiment_label,sentiment_score=get_sentiment(user_txt)
        with col2:
            st.info(f"Sentiment:{sentiment_label}")
            st.write(f"Sentiment:{sentiment_score:.3f}")

        category=detct_category(user_txt)
        with col3:
            st.info(f"category:{category}")

        st.divider()

        st.subheader('Why This Prediction Comes..')

        shap_values=explainer.shap_values(X_input)
        feature_names=tfidf.get_feature_names_out()

        shap_df=pd.DataFrame({
            'word':feature_names,
            'shap_value':shap_values[0]
        })
        shap_df=shap_df[shap_df['shap_value'] !=0]
        shap_df=shap_df.sort_values('shap_value',key=abs,ascending=False).head(15)
        st.dataframe(shap_df)
        st.toast('Prediction Complete')
















