import torch
import torch.nn as nn
from data import vocab_size, get_batch
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_lookup_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, input_token_ids):
        logits = self.vocab_lookup_table(input_token_ids)
        return logits

model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # learning rate is 0.001

#training step
for iter in range(5000):
    x,y = get_batch("train") # has shape (32,64)

    logits = model(x) #has shape (32, 64, 65)


    #convert logits to prob using softmax



    #calculate the negative log of the prob assigned to correct answer


    #original logits are (32,64,65)
    #reshape combine batch and sequence number so we get (2048, 65)


    B, T, C = logits.shape

    logits = logits.reshape(B * T, C) # (2048, 65)

    y = y.reshape(B*T) # (2048)

    loss = F.cross_entropy(logits, y) # (1) -> represents avg loss across all 2048 tokens in our batch
    print("Loss: ", loss.item())
    optimizer.zero_grad() # reset the gradients to zero for each paramter

    loss.backward() #  -> pytorch stores the gradients in the .grad attirvute of each paramter

    optimizer.step() # update the model parameters






#validation step
#telling putorch not to track the gradients so we can save memory
with torch.no_grad():
    losses = []
    for i in range(100):
        x,y = get_batch("val")
        logits = model(x)
        B, T, C = logits.shape
        logits = logits.reshape(B * T, C)
        y = y.reshape(B*T)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
    print("Validation loss: ", sum(losses) / len(losses))

def gen

