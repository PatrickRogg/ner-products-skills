import torch
from skweak.base import SpanAnnotator
from spacy.tokens import Span
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead

from config import PROD_LABEL, OTHER_LABEL, SKILL_LABEL

PRODUCTS = ['Produkt', 'Komponente', 'Komponenten', 'System', 'Systeme', 'Software', 'Technik', 'Gerät', 'Geräte',
            'Geräten' 'Werkzeug', 'Werkzeuge', 'Anlage', 'Anlagen', 'Elektro', 'Elektronik', 'Motor', 'Motoren',
            'Service', 'Kunststoff', 'Rohstoffe']
SKILLS = ['Fähigkeit', 'Softwareentwicklung', 'Maschinenbau']


class MaskedLanguageModelAnnotator(SpanAnnotator):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    model = AutoModelWithLMHead.from_pretrained("bert-base-german-cased").to(device)
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    def __init__(self):
        super(MaskedLanguageModelAnnotator, self).__init__("product_annotator")

    def find_spans(self, doc):
        for chunk in doc.noun_chunks:
            label = self.get_label(chunk, doc.text)
            if label != OTHER_LABEL:
                yield chunk.start, chunk.end, label

    def get_label(self, text_chunk: Span, text: str) -> bool:
        mask_token = self.tokenizer.mask_token
        cleaned_chunk_text = text_chunk.text.replace('(', '').replace(':', '')
        masked_sentence = text.replace(cleaned_chunk_text, mask_token, 1)

        if mask_token not in masked_sentence:
            return False

        predictions = self.fill_mask(masked_sentence)

        acc_score = 0
        top_predictions = []

        for prediction in predictions:
            if acc_score > 0.6:
                break

            predicted_word = prediction['token_str']
            acc_score += prediction['score']
            top_predictions.append({'word': predicted_word, 'score': prediction['score']})

        label = OTHER_LABEL

        for prediction in top_predictions:
            if prediction['word'] in PRODUCTS:
                label = PROD_LABEL
                break

            if prediction['word'] in SKILLS:
                label = SKILL_LABEL
                break

        return label
