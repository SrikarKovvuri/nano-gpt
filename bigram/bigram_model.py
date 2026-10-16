
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import encode, decode

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_lookup_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, input_token_ids):
        logits = self.vocab_lookup_table(input_token_ids)
        return logits

    def generate(self, prompt, max_new_tokens):
        #context is just 

        ids = encode(prompt) # list of ids
        context = torch.tensor([ids]) # (1, len(prompt))
        for i in range(max_new_tokens):

            #optimized: only process the last token
            logits = self(context[:, -1:])

            #we only care about the last one
            logits = logits[:,-1,:] # (1, 65)

            #pick the next token
            probs = F.softmax(logits, dim = -1) # we want softmax over 65 possible characters(last dimension)
            next_token = torch.multinomial(probs, num_samples = 1)

            context = torch.cat([context, next_token], dim = 1)
        
        #convert context into a list and then decode
        return decode(context[0, len(ids):].tolist())





            

