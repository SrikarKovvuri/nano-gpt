
import torch
# 1. Load the dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Find all unique characters
chars = sorted(set(text))
vocab_size = len(chars)

print("Vocabulary size:", vocab_size)

# 3. Create lookup tables
token_to_id = {}

for index, char in enumerate(chars):
    token_to_id[char] = index

id_to_token = {}

for char, index in token_to_id.items():
    id_to_token[index] = char

# 4. Encode: text -> token IDs
def encode(text):
    ids = []

    for char in text:
        ids.append(token_to_id[char])

    return ids

# 5. Decode: token IDs -> text
def decode(ids):
    chars = []

    for index in ids:
        chars.append(id_to_token[index])

    return "".join(chars)

# 6. Test the tokenizer
sample = "hello"

encoded = encode(sample)
decoded = decode(encoded)

print("Original:", sample)
print("Encoded:", encoded)
print("Decoded:", decoded)

assert decoded == sample

# 7. Convert entire dataset to a PyTorch tensor
encoded_text = encode(text)
data = torch.tensor(encoded_text)

print("Tensor SHape: ", data.shape)
print("First 20 tokens: ", data[:20])


#8. Now we need to split this tensor into 90% training and 10% for testing

length = data.size(0)
training_amt = int(0.9 * length)

training_data = data[:training_amt]
val_data = data[training_amt:]


#Now we want to split into batches. We don't want to pass everything at once bc:

#1. Self-attention is rlly rlly expensive. THis would eat up insane computation and memory

#2. We don't need the entire context. 

#3. smaller sequences enable batching. This allows the gpu to process three independent
#sequences in parallel




def get_batch(split):
    batch_size = 32
    block_size = 64

    #we don't wanna train on first 64 characters. We want to sample random subsequences.
    source = None
    if split == "train":
        source = training_data
    else:
        source = val_data
    #arguments are start, end, and how many output do you want in 1D tensor
    starts = torch.randint(0,len(source) - block_size, (batch_size,))
    #this picks random integer from 0 - 99
    x = []
    y = []
    for start in starts:
        curr_x = source[start:start+block_size]
        curr_y = source[start + 1: start + block_size + 1]

        x.append(curr_x)
        y.append(curr_y)

    x = torch.stack(x) #()
    y = torch.stack(y)

    return x, y






