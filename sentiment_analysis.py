import numpy as np
import pandas as pd
import streamlit as st
import spacy
import contractions
import re
import emoji
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore

nlp=spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])

def basic_cleaning(text):
    text=emoji.demojize(text) # Converts emoji characters into their text equivalents
    text=BeautifulSoup(text, 'html.parser').get_text() # Remove HTML tags
    text=contractions.fix(text) # Expand common english contractions with their full forms
    text=text.lower() # Convert text to lowercase
    text=re.sub(r"(.)\1{2,}", r"\1\1", text) # Correct repetitive characters (e.g., 'schooool' to 'school')
    text=re.sub(r"[@#^<>|(){}[\]\~*+=]", " ", text) # Remove less important special characters
    text=re.sub(r"\s+", " ", text) # Remove multiple consecutive white spaces
    text=text.strip() #Remove leading/trailing whitespaces
    return text

def lem(text_col, cat_col, data):
    data['cleaned']=data[text_col].apply(basic_cleaning)
    data[text_col]=data[cat_col].astype(str)+" "+data['cleaned'].astype(str) # Concatenate the 'games' column content with the 'tweets' column content
    lemm = [" ".join([token.lemma_ for token in nlp(text)]) for text in data[text_col]] # Perform lemmatization using spaCy
    return pd.Series(lemm, index=data.index)

def token(text):
    tokenizer=Tokenizer(num_words=None, filters='"#&()*+-/:;<=>@[\\]^_`{|}~\t\n"', char_level=False, oov_token=None) # Initialize the Tokenizer
    tokenizer.fit_on_texts(text) # Fit the tokenizer on the text data to build the word index (vocabulary)
    word_index=tokenizer.word_index # Get the word-to-index mapping from the fitted tokenizer
    sequence=tokenizer.texts_to_sequences(text) # Convert text to sequences based on the learned word index
    vocab_size=len(word_index)+1 # Calculate the vocabulary size
    max_seq_len=max(len(seq) for seq in sequence) # Calculate the maximum sequence length among all sequences
    padded_seq=pad_sequences(sequence, maxlen=max_seq_len, padding='post') # Pad the sequences to the maximum length
    return padded_seq, max_seq_len, word_index, vocab_size

bilstm_model = load_model("C:\\Users\\rothi\\NewBegining\\Sentiment_analysis\\bilstm_model.h5")

st.set_page_config(
    page_title="🎭 Digital Arena Sentiment Analyzer",
    page_icon="🎭",
    layout="centered",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/rathindra-narayan-hait-a0ba9015a/',
        'Report a bug': "https://rathindra.onrender.com/",
        'About': "### Digital Arena Sentiment Analyzer is an innovative tool designed to analyze conversations happening around your favorite games."
    }
)

@st.dialog("Impression")
def sentiment(data):
    with (st.container(border=True)):
        st.dataframe(data)
        data['tweets']=lem('tweets', 'games', data.copy())
        pad, _, _, _=token(data['tweets'])
        labels = ["Irrelevant", "Negative", "Neutral", "Positive"]

        with st.spinner("Analyzing sentiment..."):
            prediction=bilstm_model.predict(pad)
            pred=np.argmax(prediction)
            top_label = labels[pred]
            preds = prediction[0]  # Assuming shape is (1, 4)
            st.subheader("🎭 Predicted Sentiment")
            st.success(f"**{top_label}** with {preds[pred]*100:.2f}% confidence")
            st.subheader("Sentiment Breakdown")
            df = pd.DataFrame({"Sentiment": labels, "Confidence (%)": [round(p*100, 2) for p in preds]})
            st.bar_chart(df.set_index("Sentiment"))

        with (st.container()):
                co1, co2, co3=st.columns([1, 1, 1])
                with co2:
                    if st.button("Back", icon=':material/reply:', use_container_width=True ):
                        st.rerun()

with st.container():

    colu1, colu2, colu3=st.columns([1,4,1])

    with colu2:

        st.title("🎭 Digital Arena Sentiment Analyzer")
        st.write("######")

        with (st.container(border=True)):
            st.write("######")
            col1, col2, col3, col4=st.columns([0.2, 1, 2.5, 0.2])
            with col2:
                games=st.text_input(label="Games", icon=":material/sports_esports:", placeholder="Enter the video game name.....")
            with col3:
                tweets=st.text_area(label="Tweet/Statement", height=100, placeholder="Enter The Tweet or Statement, You Would Like to Analyze.....")
            with (st.container()):
                coll1, coll2, coll3=st.columns([1, 1, 1])
                with coll2:
                    button_disabled=not tweets or tweets.strip().isdigit()
                    if st.button("Analyze", icon=":material/theater_comedy:", use_container_width=True, disabled=button_disabled):
                        data=pd.DataFrame([[games, tweets]], columns=["games", "tweets"])

                        sentiment(data)
                st.write("######")
        st.caption("*This app predicts sentiment of statements related to video games using a BiLSTM model")
        st.write('---')