import streamlit as st
from deep_learning_pipeline import NlpPredictionPipeline

st.title('Resturant Reviews Sentiment Detection Project')
st.warning('Detects if resturant review is positive or negative in emotion')

pipeline = NlpPredictionPipeline()

st.write('')
st.write('')

input_text = st.text_area('Type your review here: ', '')
if st.button('Predict'):
    y_preds, y_probs = pipeline.predict_review_sentiment(text=input_text)
    if y_preds[0][0] == 0:
        st.error('Negative Review detected!')
        acc = "{:.2f}".format(100*(1 - y_probs[0][0]))
        st.success(f'Accuracy: {acc}%')
    else:
        st.success('Positive Review Detected!')
        acc = "{:.2f}".format(100*(y_probs[0][0]))
        st.success(f'Accuracy: {acc}%')
