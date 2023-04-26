import torch
from transformersv2 import AutoTokenizer, AutoModel

device='cpu'

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert_mlp_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
# breakpoint()

sentences = [tokenizer(f"a photo of a {c}", return_tensors="pt")['input_ids'] for c in ["No DR","mild DR", "moderate DR", "severe DR", "proliferative DR"]]
max_token_length = max([s.shape[1] for s in sentences])
sentences = [tokenizer(f"a photo of a {c}", max_length=max_token_length, padding='max_length', truncation=True, return_tensors="pt")['input_ids'] for c in ["No DR","mild DR", "moderate DR", "severe DR", "proliferative DR"]]
sentences = torch.cat(sentences, 0).to(device)

for param in biobert_mlp_model.parameters():
    param.requires_grad = False

biobert_mlp_model.embeddings.prompt_tokens.requires_grad = True

text_features = biobert_mlp_model(sentences).pooler_output

print(text_features.shape)

breakpoint()
# print(sentences.shape)
# breakpoint()
# outputs = torch.cat([biobert_mlp_model(s.to(device)).pooler_output for s in sentences])

# print(outputs.shape)