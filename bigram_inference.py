import torch
from bigram_model import BigramLanguageModel
from data import vocab_size

#init model -> just has random parameters right now
model = BigramLanguageModel(vocab_size)

#load the trained weights
model.load_state_dict(
    torch.load("bigram.pt", map_location="cpu", weights_only=True)
)

model.eval()

prompt = input("Enter your prompt: ")

#run inference without tracking gradients
with torch.inference_mode():
    response = model.generate(prompt, 200)


print("\n")
print(response)

