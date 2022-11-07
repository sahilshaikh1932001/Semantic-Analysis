# -*- coding: utf-8 -*-
"""semantic similarity analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QWI13_njqBrBSBwFe3Ypwl4RiHD-5nTh
"""

# importing necessary libraries
import pandas as pd
import numpy as np

# loading data set
df = pd.read_csv(r'/content/drive/MyDrive/Precily task/Precily_Text_Similarity.csv')

df.head()

pip install transformers

pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util          # I imported SentenceTransformer for loading-pre trained model

model = SentenceTransformer('stsb-mpnet-base-v2')

sentence1 = "currently i am studying because tomorrow is my exam"
sentence2 = "I want to study as my exam will start soon"

# encode sentences to get their embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", cosine_scores.item())

sentences1 = df.iloc[:15,0].values     # as we have so much ammount of rows that consumes so much time so for timing i am taking 15 rows
sentences2 = df.iloc[:15,-1].values
# encode list of sentences to get their embeddings
embedding1 = model.encode(sentences1, convert_to_tensor=True)
embedding2 = model.encode(sentences2, convert_to_tensor=True)
# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("Sentence 1:", sentences1[i])
        print("Sentence 2:", sentences2[j])
        print("Similarity Score:", cosine_scores[i][j].item())
        print()

sentences1 = df.iloc[:15,0].values     # as we have so much ammount of rows that consumes so much time so for timing i am taking 15 rows
sentences2 = df.iloc[:15,-1].values
# encode list of sentences to get their embeddings
embedding1 = model.encode(sentences1, convert_to_tensor=True)
embedding2 = model.encode(sentences2, convert_to_tensor=True)
# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

score = []
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("Sentence 1:", sentences1[i])
        print("Sentence 2:", sentences2[j])
        print("Similarity Score:", cosine_scores[i][j].item())
        score.append(cosine_scores[i][j].item())
        print()

len(score)

print(score)



