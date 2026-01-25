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