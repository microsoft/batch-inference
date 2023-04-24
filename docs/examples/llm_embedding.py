from typing import Tuple

import torch
from flask import Flask, jsonify, request
from transformers import BertModel, BertTokenizer

from batch_inference import batching
from batch_inference.batcher.bucket_seq_batcher import BucketSeqBatcher


@batching(
    batcher=BucketSeqBatcher(padding_tokens=[0, 0], buckets=[25, 50, 100, 200, 500], tensor='pt'),
    max_batch_size=5,
)
class BertEmbeddingModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def preprocessing(self, text) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_text = self.tokenizer.encode_plus(
            text, return_attention_mask=True, return_tensors="pt"
        )
        return encoded_text["input_ids"], encoded_text["attention_mask"]

    def predict_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            embedding = outputs[0]
        return embedding


bert_host = BertEmbeddingModel.host()
bert_host.start()
app = Flask(__name__)


@app.route("/embed", methods=["POST"])
def embed():
    text = request.json["text"]
    input_ids, attention_mask = bert_host.model_obj.preprocessing(text)
    embedding = bert_host.predict(input_ids, attention_mask)
    return jsonify({"embedding": embedding.tolist()})


if __name__ == "__main__":
    # send warm up query to verify
    input_ids, attention_mask = bert_host.model_obj.preprocessing("hello world")
    embedding = bert_host.predict(input_ids, attention_mask)
    print(embedding)
    app.run(threaded=True)
