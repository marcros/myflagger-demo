import streamlit as st
from annotated_text import annotated_text, annotation
from model import PipadiTransfomer
import requests
import os
import zipfile

"""
# MyFlagger - Automatic Information Extraction
Demo of our NLP model to automatically extract entities from text.
"""


def prepare_annotation(eval_subtokens, predictions):
    # Get only meaningful tokens and their predictions
    tokens, preds = [], []
    i = 0

    for stk, pred in zip(eval_subtokens, predictions):
        if '[CLS]' in stk or '[SEP]' in stk:
            continue
        if '##' in stk:
            tokens[i - 1] = tokens[i - 1] + stk.strip('##')
            continue
        tokens.append(stk)
        preds.append(pred)
        i += 1

    # Convert into required input by annotated_text
    web_input = []
    for i, (s, a) in enumerate(zip(tokens, preds)):
        if a == 'O':
            if i == 0:
                web_input.append(s + ' ')
            else:
                web_input.append(' ' + s + ' ')
        else:
            web_input.append((s, a))

    return web_input


def download_model():
    folder_name = 'flagger_model'

    if os.path.exists(folder_name):
        pass
    else:
        r = requests.get('https://www.dropbox.com/sh/lypndyn08t4jxad/AAALbP7tnoY5meKXcuduJLuUa?dl=1',
                         allow_redirects=True)
        open('{}.zip'.format(folder_name), 'wb').write(r.content)

        with zipfile.ZipFile('{}.zip'.format(folder_name), 'r') as zip_ref:
            zip_ref.extractall(folder_name)
        os.remove('{}.zip'.format(folder_name))


download_model()
#with st.echo():
sentence = st.text_input('Input your sentence here:')

# Instantiate PipadiTransformer object
pipadi_model = PipadiTransfomer('flagger_model', 'distilbert-base-multilingual-cased', 220, 16)

# Tokenize input sentence
subtokens = pipadi_model.tokenize(sentence)
# Predict labels
pred = pipadi_model.predict(num_labels=17)

# Get annotation input
"""
Annotated text:
"""
proto = prepare_annotation(subtokens, pred)
annotated_text(*proto)

st.text("")

#with st.echo():
#    annotated_text(*proto)



