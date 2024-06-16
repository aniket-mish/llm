# LLMs

In this project, I'm implementing LLM from scratch, one concept at a time.

## Tokenizer

Tokenization means converting the raw text as a string to some sequence of integers according to some vocabulary of possible elements

To get started I'm using the tiny_shakespeare dataset. a simple implementation of a tokenizer could be at the character level or word level.

### Reading the data

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("number of characters: ", len(text))
print(text[:200])
```

```
number of characters:  1115394
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor
```

Get a list of all the unique characters in our dataset.

```python
vocab = sorted(list(set(text)))
```

## Converting text to tokens

Let's build a simple character-level tokenizer with encode and decode methods.

```python
class tokenizer:
    def __init__(self, vocab):
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }

    def encode(self, text):
        ids = [self.stoi[c] for c in text]
        return ids

    def decode(self, ids):
        text = ''.join([self.itos[i] for i in ids])
        return text
```

initiating the tokenizer and trying our encoder and decoder.

```python
tokenizer = Tokenizer(text)
text = """we are building an agi"""
print(tokenizer.encode(text))
print(tokenizer.decode(ids))
```

```python
[0, 5, 18, 16, 4, 5, 18, 7, 8, 12, 10, 11, 12, 17, 14, 18, 16, 17, 18, 16, 14, 12]
we are building an agi
```

we can also add special tokens like `|end_of_text|`, `|reserved_special_token_0|`

I'm using tiktoken which uses a byte pair encoding(BPE) algorithm for tokenization. Andrej Karpathy has a clean implementation of the BPE algorithm. 

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
```

we can encode and decode

```python
text = "we are building an agi"
ids = enc.encode(text, allowed_special={ "<|end_of_text|>" })
print(ids)
print(enc.decode(ids))
```

```
[854, 553, 6282, 448, 1017, 72]
we are building an agi
```

Andrej has a [YouTube video](https://www.youtube.com/watch?v=zduSFxRajkE) on the BPE.

to handle an unknown word BPE breaks down the word into characters.
