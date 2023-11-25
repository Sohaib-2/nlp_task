import streamlit as st
import nltk
from nltk import ngrams
import spacy

# Download NLTK resources
nltk.download('punkt')

def generate_ngrams(text, n):
    words = nltk.word_tokenize(text)
    n_grams = list(ngrams(words, n))
    return n_grams

# Streamlit web application
def main():
    st.title("N-gram and Entity Recognition App")

    # Input text
    input_text = st.text_area("Enter your text:")

    # N-gram generation
    n_value = st.slider("Select the 'n' value for N-grams:", min_value=1, max_value=5)
    if st.button("Generate N-grams"):
        if input_text:
            ngrams_result = generate_ngrams(input_text, n_value)
            st.write(f"{n_value}-grams:", ngrams_result)
        else:
            st.warning("Please enter some text.")
