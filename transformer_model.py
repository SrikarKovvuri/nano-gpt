import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_data import encode, decode


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size=64, n_embd=64, num_layers=3, num_heads=4):
        super().__init__()

        self.block_size = block_size

        #lookup table for each character
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        #lookup table for each position
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #create transformer blocks
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(n_embd, block_size, num_heads)
            )

        #final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        #project each token representation into vocab logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_token_ids, kv_caches=None):
        B, T = input_token_ids.shape

        #initialize empty cache slots for each transformer block
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        #number of tokens already cached
        past_length = 0

        if kv_caches[0] is not None:
            past_length = kv_caches[0][0][0].shape[1]

        #make sure we don't exceed the context window
        if past_length + T > self.block_size:
            raise ValueError("Context exceeds block_size")

        #get the embeddings for each token
        token_embeddings = self.token_embedding_table(input_token_ids)
        # (B, T, n_embd)

        #get the position embeddings for each position
        positions = torch.arange(
            past_length,
            past_length + T,
            device=input_token_ids.device
        )

        position_embeddings = self.position_embedding_table(positions)
        # (T, n_embd)

        #add the token and position embeddings
        x = token_embeddings + position_embeddings
        # (B, T, n_embd)

        #apply the transformer blocks
        new_all_caches = []

        for block, cache in zip(self.blocks, kv_caches):
            x, new_cache = block(x, kv_caches=cache)
            new_all_caches.append(new_cache)

        #final layer norm
        x = self.ln_f(x)

        #project each token representation into vocab logits
        logits = self.lm_head(x)
        # (B, T, vocab_size)

        return logits, new_all_caches

    def generate(self, prompt, max_new_tokens):
        ids = encode(prompt) # list of token IDs

        if not ids:
            raise ValueError("Prompt must contain at least one token")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")

        if max_new_tokens == 0:
            return ""

        #we currently only support generation within block_size
        if len(ids) + max_new_tokens - 1 > self.block_size:
            raise ValueError("Prompt + generation exceeds block_size")

        #convert ids to tensor
        context = torch.tensor(
            [ids],
            dtype=torch.long,
            device=self.token_embedding_table.weight.device
        )

        #prefill: process the entire prompt and populate KV caches
        logits, kv_caches = self(context, kv_caches=None)

        for i in range(max_new_tokens):

            #we only care about the last token's prediction
            next_logits = logits[:, -1, :] # (1, vocab_size)

            #sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # (1, 1)

            #concatenate next token to context
            context = torch.cat([context, next_token], dim=1)

            #decode: process only the newly generated token
            #skip after the final token because no next prediction is needed
            if i < max_new_tokens - 1:
                logits, kv_caches = self(
                    next_token,
                    kv_caches=kv_caches
                )

        #return only generated characters
        return decode(context[0, len(ids):].tolist())

class TransformerHead(nn.Module):
    #init always defines the learanble layers and parameters
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias = False) # (B, T, head_size)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        #create a mask
        # self.register buffer is used so we can move it to cuda if needed. torch.tril upper diagonal is zeroes. torch.ones is justa  block of ones.
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    #forward mererely uses those layers and parameters to perfrom computations on the input
    def forward(self, x, kv_cache = None):
        B, T, C = x.shape
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        #everything above is (B, T, head_size)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim = 1)
            v = torch.cat([past_v, v], dim = 1)
        
        new_cache = (k,v)


        #compute attention scores
        #S = QK^T
        scores = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        #scale the scores by the square root of the head size
        scores = scores * (k.shape[-1]) ** -0.5
        #mask the scores. We slice the mask to the same size as the ampunt of input tokens
         #mask future tokens
        total_key_length = k.shape[1]
        past_length = total_key_length - T

        mask = self.mask[
            past_length:past_length + T,
            :total_key_length
        ]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        #softmax the scores
        weights = F.softmax(scores, dim=-1) # (B, T, T)
        #apply the weights to the values
        out = weights @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out, new_cache




class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, block_size):
        super().__init__()

        assert n_embd % num_heads == 0
        head_size = n_embd // num_heads

        #create the heads
        self.heads = nn.ModuleList()

        for i in range(num_heads):
            self.heads.append(
                TransformerHead(n_embd, head_size, block_size)
            )

        #project concatenated heads back into embedding space
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, kv_caches = None):
        head_outputs = []
        new_caches = []

        if kv_caches is None:
            kv_caches = [None] * (len(self.heads))
        
        for head, cache in zip(self.heads, kv_caches):
            out, new_cache = head(x, kv_cache=cache)

            head_outputs.append(out)
            new_caches.append(new_cache)

        #concatenate along the feature dimension
        out = torch.cat(head_outputs, dim=-1)

        #mix information across the heads
        out = self.proj(out)

        return out, new_caches



class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, block_size, num_heads):
        super().__init__()

        #multi head attention
        self.attention = MultiHeadAttention(num_heads, n_embd, block_size)

        #ffn
        self.feed_forward = FeedForward(n_embd)

        #layer_norm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, kv_caches = None):
        #layer norm -> attention -> residual
        #x_new = x_old + attention(layer_norm(x_old))
        attention_out, new_caches = self.attention(
            self.ln1(x),
            kv_caches = kv_caches
        )
        x = x + attention_out

        #layer norm -> ffn -> residual
        #x_new = x_old + ffn(layer_norm(x_old))
        x = x + self.feed_forward(self.ln2(x))

        return x, new_caches





        
