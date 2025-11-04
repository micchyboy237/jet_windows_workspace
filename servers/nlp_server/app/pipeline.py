# app/pipeline.py
import stanza
from typing import List
from .config import settings

class NLPProcessor:
    def __init__(self):
        self._nlp = self._initialize_pipeline()

    def _initialize_pipeline(self):
        stanza.download(lang=settings.language,
                        model_dir=settings.model_dir,
                        processors=settings.processors,
                        verbose=False)
        # build pipeline
        return stanza.Pipeline(lang=settings.language,
                                processors=settings.processors,
                                model_dir=settings.model_dir,
                                use_gpu=settings.use_gpu,
                                verbose=False)

    def annotate(self, texts: List[str]) -> List[dict]:
        """
        Annotate a list of texts and return list of dict results.
        """
        docs = [self._nlp(text) for text in texts]
        results = []
        for doc in docs:
            doc_res = {
                "sentences": [
                    {
                        "text": sent.text,
                        "tokens": [token.text for token in sent.tokens],
                        "words": [
                            {
                                "text": w.text,
                                "lemma": w.lemma,
                                "pos": w.pos,
                                "ner": getattr(w, "ner", None)
                            }
                            for w in sent.words
                        ]
                    }
                    for sent in doc.sentences
                ]
            }
            results.append(doc_res)
        return results

# Create a single processor instance (singleton)
nlp_processor = NLPProcessor()