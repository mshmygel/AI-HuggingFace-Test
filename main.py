from transformers import pipeline

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline("summarization", model="facebook/bart-large-cnn")
response = model("text to summarize")
print(response)