# from skweak.base import SpanAnnotator
# from spacy.tokens import Span
#
# from config import PROD_LABEL, OTHER_LABEL, SKILL_LABEL
#
#
# class GoogleNLPAnnotator(SpanAnnotator):
#
#     def __init__(self):
#         super(GoogleNLPAnnotator, self).__init__("product_annotator")
#
#     def find_spans(self, doc):
#         for chunk in doc.noun_chunks:
#             if label != OTHER_LABEL:
#                 yield chunk.start, chunk.end, label
