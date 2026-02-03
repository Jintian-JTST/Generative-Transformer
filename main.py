import torch
import torch.nn as nn
from torch.nn import functional as F
device = "xpu"
batch_size = 4
block_size = 8  # context length for predictions
batch_size = 32
max_iters = 10000
eval_interval = 500
n_embd = 32

with open(r"dataset/01.txt", "r", encoding='utf-8') as file:
    content = file.read()

'''print(len(content))
print(content[:100])'''  # Print the first 100 characters to verify content

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
data = torch.tensor(encode(content), dtype=torch.long, device=device)

'''print(encode("hello world"))
print(decode(encode("hello world")))'''

'''print(data.shape, data.dtype)
print(data[:100])'''  # Print the first 100 encoded integers to verify encoding

# Split the data into training and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Demonstrate how to create input-target pairs for training
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
'''for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context.tolist()} the target: {target}")'''

torch.manual_seed(42)
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

x, y = get_batch('train')
'''print("inputs:")
print(x)
print("targets:")
print(y)'''


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx
    

'''logits, loss = model(x, y)
print("logits shape:", logits.shape)
print("loss:", loss)
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))  # Decode the predicted tokens for verification

'''    
# 假设你之前已经定义了 vocab_size 和获取了 x, y
model = BigramLanguageModel(vocab_size)
m = model.to(device)
logits, loss = m(x, y)
print("logits shape:", logits.shape)
print("loss:", loss)
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))  # Decode the predicted tokens for verification

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for steps in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_interval == 0:
        print(f"step {steps}: loss {loss.item()}")
        print("----")
        print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))  # Decode the predicted tokens for verification