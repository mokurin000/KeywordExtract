import os
from logging import INFO
from string import ascii_letters, digits

import jieba
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from flair.embeddings import TransformerDocumentEmbeddings

from keywordextract.keywords import KEYWORDS


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

roberta = TransformerDocumentEmbeddings("hfl/chinese-roberta-wwm-ext-large")
kw_model = KeyBERT(model=roberta)

jieba.default_logger.setLevel(INFO)
for word in KEYWORDS:
    jieba.add_word(word)

with open("data/chineseStopWords.txt", encoding="utf-8") as f:
    stopwords = set(f.read().split("\n"))


def tokenize_zh(text):
    words = jieba.cut(
        text,
    )
    return list(
        word
        for word in words
        if word not in stopwords
        if word.isalpha()
        if not all(map(lambda c: c in ascii_letters + digits, word))
    )


vectorizer = CountVectorizer(tokenizer=tokenize_zh, token_pattern=None)
