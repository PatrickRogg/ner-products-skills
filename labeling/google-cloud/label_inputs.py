import json
import os
from os import listdir

from analyse_entities import analyze_entities
from config import PROD_LABEL
from utils.is_product_page import is_product_page

filenames = [f for f in listdir(f'..{os.sep}data{os.sep}inputs')]
seen_sentences = set()

for filename in filenames:
    with open(f'../data/inputs/{filename}') as json_file:
        pages = json.load(json_file)

    labeled_data = []

    for page in pages:
        if not is_product_page(page['url'], page['title']):
            print('Not Product page')
            continue

        print(f'Product page: {page["url"]} - {page["title"]}')
        body: str = page['body']
        sentences = split_text_into_sentences(page['body'])

        for sentence in sentences:
            if sentence in seen_sentences:
                print('seen')
                continue

            seen_sentences.add(sentence)
            labeled_sentence = analyze_entities(sentence)

            if PROD_LABEL in labeled_sentence['labels']:
                labeled_data.append(labeled_sentence)

    with open(f'../data/google-nlp/{filename}', 'w') as outfile:
        json.dump({
            'data': labeled_data
        }, outfile, sort_keys=True, indent=4)
