with open(r"dataset/01.txt", "r", encoding='utf-8') as file:
    content = file.read()

print(len(content))
print(content[:100])  # Print the first 100 characters to verify content

chars=sorted(list(set(content)))
vocab_size=len(chars)
print("Vocab size:", vocab_size)
print("Characters:", ''.join(chars))

# Create mappings from characters to integers and vice versa
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join(itos[i] for i in l)

print(encode("hello world"))
print(decode(encode("hello world")))

import torch
data = torch.tensor(encode(content), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])  # Print the first 100 encoded integers to verify encoding

# Split the data into training and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Demonstrate how to create input-target pairs for training
block_size = 8  # context length for predictions
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context.tolist()} the target: {target}")

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
batch_size = 4
x, y = get_batch('train')
print("inputs:")
print(x)
print("targets:")
print(y)