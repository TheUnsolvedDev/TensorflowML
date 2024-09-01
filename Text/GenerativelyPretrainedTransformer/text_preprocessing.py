import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
# import nltk
import string
import re
import json
import os

from config import *

# nltk.download('stopwords')


class DataCleaning:
    def __init__(self, path=''):
        self.path = path
        
        with open(self.path, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                line = self.clean(line)
                new_lines.append(line)

        self.data = ' '.join(new_lines)

        # self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def to_lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        punctuation = string.punctuation.replace('.', '').replace("'", '')
        return text.translate(str.maketrans('', '', punctuation))

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def strip_whitespace(self, text):
        return text.strip()

    def remove_stopwords(self, text):
        words = text.split()
        cleaned_text = ' '.join(
            word for word in words if word not in self.stop_words)
        return cleaned_text

    def replace_words(self, text):
        text = text.replace('<sos>', '')
        text = text.replace('<eos>', '')
        text = text.replace('<nl>', '.')
        text = text.replace('.', ' .')
        for i in range(3):
            text = text.replace(' . .','. ')
        return text

    def remove_bracketed_text(self, text):
        return re.sub(r'\[.*?\]', '', text)

    def clean(self, text):
        text = self.to_lowercase(text)
        text = self.replace_words(text)
        text = self.remove_bracketed_text(text)
        text = self.remove_punctuation(text)
        text = self.strip_whitespace(text)
        # text = self.remove_stopwords(text)
        return text


    def create_dataset(self):
        self.data = self.data.split('.')
        self.new_data = [i for i in self.data if len(i.split(' ')) > 4]
        return self.new_data



if __name__ == "__main__":
    tokenizer = DataCleaning('Dataset/reddit_short_stories_cleaned.txt')
    data = tokenizer.create_dataset()