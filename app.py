# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 00:09:47 2022

@author: shaik
"""


import streamlit as st
import numpy as np 
import time

# adding title
st.title('semantic analysis')

# adding image 
st.image("Semantic Analysis.jpg",width = 800)


# Markdown
st.markdown("#### Semantic Similarity, or Semantic Textual Similarity, is a task in the area of Natural Language Processing (NLP) that scores the relationship between texts or documents using a defined metric.There have been a lot of approaches for Semantic Similarity. The most straightforward and effective method now is to use a powerful model (e.g. transformer) to encode sentences to get their embeddings and then use a similarity metric (e.g. cosine similarity) to compute their similarity score.")


sentence_1 = st.text_input(label = 'enter the 1st sentence ', label_visibility="visible")

sentence_2 = st.text_input(label = 'enter the 2nd sentence ', label_visibility="visible")


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('stsb-roberta-large')

# encode sentences to get their embeddings
embedding1 = model.encode(sentence_1, convert_to_tensor=True)
embedding2 = model.encode(sentence_2, convert_to_tensor=True)

# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)


if st.button("Predict"):
    
    progress = st.progress(0)    # this is for progress bar
    for i in range(100):
        time.sleep(0.001)
        progress.progress(i+1)
        
    st.success('similarity score is :{}'.format(cosine_scores.item()))
    
   