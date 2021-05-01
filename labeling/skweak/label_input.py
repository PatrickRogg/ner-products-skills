import json
import os
from os import listdir

import skweak
import spacy
import spacy.training
from skweak.aggregation import MajorityVoter
from skweak.base import CombinedAnnotator
from spacy.tokens import Doc
from spacy.training import offsets_to_biluo_tags

from config import PROD_LABEL
from labeling.skweak.labeling_functions.masked_lm_labeling_function import MaskedLanguageModelAnnotator
from utils.is_product_page import is_product_page


def build_doc_from(text: str):
    nlp = spacy.load("de_core_news_sm")
    docs = []

    # breaks sentences into separate documents
    for sent in nlp(text).sents:
        docs.append(nlp(sent.text))

    return docs


def annotate(doc):
    combined_annotator = CombinedAnnotator()
    combined_annotator.add_annotator(MaskedLanguageModelAnnotator())

    return list(combined_annotator.pipe(doc))


def remove_docs_without_labels(docs):
    return [d for d in docs if any([v for (k, v) in d.spans.items()])]


def aggregate_labels(docs):
    if len(docs) == 0:
        return docs

    voter = MajorityVoter("maj_voter", labels=[PROD_LABEL], sequence_labelling=True)

    for d in docs:
        voter(d)

    hmm = skweak.aggregation.HMM('hmm', [PROD_LABEL])
    return hmm.fit_and_aggregate(docs)


def replace_ner_spans(doc: Doc, source: str):
    spans = []
    if source in doc.spans:
        for span in doc.spans[source]:
            spans.append(span)
    doc.ents = tuple(spans)

    return doc


def build_json_from(docs):
    labeled_sentences = []
    for i, d in enumerate(docs):
        d = replace_ner_spans(d, 'product_annotator')
        json_doc = spacy.training.docs_to_json([d])
        words = [w.text for w in d]
        labels = offsets_to_biluo_tags(d, json_doc['paragraphs'][0]['entities'])
        text = json_doc['paragraphs'][0]['raw']

        labeled_sentences.append({
            'text': text,
            'words': words,
            'labels': labels
        })

        if i > 0 and i % 1000 == 0:
            print("Converted documents:", i)

    return {
        'labeled_sentences': labeled_sentences
    }


def label(text: str):
    doc = build_doc_from(text)
    labeled_docs = annotate(doc)
    labeled_docs = remove_docs_without_labels(labeled_docs)
    aggregated_docs = aggregate_labels(labeled_docs)

    return build_json_from(aggregated_docs)


def label_input_files():
    labeled_texts = []

    input_dir = f'..{os.sep}..{os.sep}data{os.sep}inputs{os.sep}'
    output_dir = f'..{os.sep}..{os.sep}data{os.sep}labeled-data{os.sep}'
    filenames = [f for f in listdir(input_dir)]
    seen_texts = set()

    for filename in filenames:
        with open(f'{input_dir}{filename}') as json_file:
            pages = json.load(json_file)

        for page in pages:
            if 'body' not in page:
                print('No body')
                continue

            if not is_product_page(page):
                print('Not Product page')
                continue

            if page['body'] in seen_texts:
                print('Text already seen')
                continue

            seen_texts.add(page['body'])
            labeled_text = label(page['body'])

            if len(labeled_text['labeled_sentences']) > 0:
                labeled_texts.append(labeled_text)

        with open(f'{output_dir}{filename}', 'w') as outfile:
            json.dump({
                'data': labeled_texts
            }, outfile, sort_keys=True, indent=4)

    return labeled_texts


if __name__ == "__main__":
    label_input_files()
