import os
import tiktoken
import numpy as np

# Read the medical textbook data
input_file_path = os.path.join(os.path.dirname(__file__), 'med_textbook.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Length of dataset in characters: {len(data):,}")

# Split into train and validation sets (90% train, 10% val)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print(f"Train split: {len(train_data):,} characters")
print(f"Val split: {len(val_data):,} characters")

# Encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("Data preparation complete!")
print(f"train.bin has {len(train_ids):,} tokens")
print(f"val.bin has {len(val_ids):,} tokens")
