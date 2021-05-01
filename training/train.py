import json
import os
from os import listdir

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, pipeline, TrainingArguments, \
    Trainer

from config import PROD_LABEL, SKILL_LABEL, OTHER_LABEL
from training.entity_dataset import EntityDataset

entity_to_id = {f'B-{PROD_LABEL}': 0, f'L-{PROD_LABEL}': 1, f'I-{PROD_LABEL}': 2, f'U-{PROD_LABEL}': 3,
                f'B-{SKILL_LABEL}': 4, f'L-{SKILL_LABEL}': 5, f'I-{SKILL_LABEL}': 6,
                f'U-{SKILL_LABEL}': 7, OTHER_LABEL: 8}

configuration = AutoConfig.from_pretrained('distilbert-base-multilingual-cased')
configuration.num_labels = len(entity_to_id)
configuration.id2label = {v: k for k, v in entity_to_id.items()}
configuration.label2id = entity_to_id
model = AutoModelForTokenClassification.from_config(configuration)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")


def build_ner_model():
    ner = pipeline('ner', model=model, tokenizer=tokenizer)
    return ner


def encode_entities(words, entities, encodings):
    labels = [[entity_to_id[entity] for entity in doc] for doc in entities]
    encoded_labels = []

    for doc_words, doc_labels, doc_encoding in zip(words, labels, encodings['input_ids']):
        i = 0
        j = 1
        doc_enc_labels = np.ones(len(doc_encoding), dtype=int) * -100
        tokens = tokenizer.convert_ids_to_tokens(doc_encoding)
        word = ''

        while i < len(doc_words) and j < len(tokens):
            if len(doc_words[i].strip()) == 0:
                i += 1
                continue

            token = tokens[j].replace('##', '')
            word += token
            doc_enc_labels[j] = doc_labels[i]

            if word == doc_words[i]:
                word = ''
                i += 1
            j += 1

        encoded_labels.append(doc_enc_labels)

    return encoded_labels


def get_words_and_labels():
    labeled_data_folder = f'data{os.sep}labeled-data{os.sep}'
    filenames = [f for f in listdir(labeled_data_folder)]
    words = []
    labels = []

    for filename in filenames:
        with open(f'{labeled_data_folder}{filename}') as json_file:
            dataset = json.load(json_file)['data']

        for data in dataset:
            for labeled_sentence in data['labeled_sentences']:
                if len(labeled_sentence) > 0:
                    words.append(labeled_sentence['words'])
                    labels.append(labeled_sentence['labels'])

    return words, labels


def build_trainer():
    training_args = TrainingArguments(
        output_dir='results',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=30,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='logs',  # directory for storing logs
    )

    words, labels = get_words_and_labels()
    train_texts, val_texts, train_entities, val_entities = train_test_split(words, labels, test_size=.2)
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                              truncation=True)
    train_labels = encode_entities(train_texts, train_entities, train_encodings)
    val_labels = encode_entities(val_texts, val_entities, val_encodings)

    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=EntityDataset(train_encodings, train_labels),
        eval_dataset=EntityDataset(val_encodings, val_labels)

    )


def train():
    ner = build_ner_model()
    trainer = build_trainer()
    trainer.train()
    trainer.evaluate()

    print(ner('Ich kaufe gerne ein Iphone'))

    model.save_pretrained(f'..{os.sep}models')
