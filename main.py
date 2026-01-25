with open(r"dataset/01.txt", "r", encoding='utf-8') as file:
    content = file.read()

print(len(content))
print(content[:100])  # Print the first 100 characters to verify content

chars=sorted(list(set(content)))
vocab_size=len(chars)
print("Vocab size:", vocab_size)
print("Characters:", ''.join(chars))