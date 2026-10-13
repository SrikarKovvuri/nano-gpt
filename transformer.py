
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))
print("First 500 characters:")
print(text[:500])
