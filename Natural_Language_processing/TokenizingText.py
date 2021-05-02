# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:37:29 2021

@author: Tonny
#Tokenizing text to in NLP
"""

#read data file from URL
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv \
    -O /tmp/bbc-text.csv

#import required modules
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
# Convert it to a Python list and paste it here
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


## Obtain the sentences and labels from the downloaded files
sentences = []
labels = []
with open("/tmp/bbc-text.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    labels.append(row[0])
    sentences.append(row[1])

labels.pop(0)
sentences.pop(0)

print(len(sentences))
print(sentences[0])


#define the sentences tokenizer
tokenizer = Tokenizer( oov_token="unknown_words")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))

# Expected output
# 29714
#Process the tokenizer
sequences = tokenizer.texts_to_sequences(sentences)
print(len(sequences))
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)

#Tokenize the labels

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)

print(label_seq)
print(label_word_index)