import torch
from transformers import BertTokenizer

print("PyTorch version:", torch.__version__)
print("Torch backend OK:", torch.backends.mps.is_available())  # Apple Silicon only
tokenizer = BertTokenizer.from_pretrained("skt/kobert-base-v1")
print("Tokenizer loaded!")
