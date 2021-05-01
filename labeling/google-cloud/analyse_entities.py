from google.cloud import language_v1
from google.oauth2 import service_account

from config import PROD_LABEL

credentials = service_account.Credentials.from_service_account_file("../../service-key.json")
client = language_v1.LanguageServiceClient(credentials=credentials)
type_ = language_v1.Document.Type.PLAIN_TEXT
language = "de"
encoding_type = language_v1.EncodingType.UTF8


def analyze_entities(sentence):
    words = sentence.split(' ')
    labels = []
    document = {"content": sentence, "type_": type_, "language": language}
    response = client.analyze_entities(request={'document': document, 'encoding_type': encoding_type})
    consumer_good_entities = []

    for entity in response.entities:
        if language_v1.Entity.Type(entity.type_).name == 'CONSUMER_GOOD':
            consumer_good_entities.append(entity)

    for i, word in enumerate(words):
        found = False

        for entity in consumer_good_entities:
            if word == entity.name:
                found = True
                break

            if entity.name in word:
                found = True
                words[i] = entity.name
                break

        if found:
            labels.append(PROD_LABEL)
        else:
            labels.append('O')

    return {
        'sentence': sentence,
        'words': words,
        'labels': labels
    }

