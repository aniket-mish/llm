# Building an LLM

I'm implementing an LLM from scratch, one component at a time.

## Setup

1. Download and install Miniforge

https://github.com/fastai/fastsetup/blob/master/setup-conda.sh

2. Create new virtual environment

```bash
conda create -n llms python=3.10
```

```bash
conda activate llms
```

3. Install required python packages

```bash
pip install -r requirements.txt
```

1. Building
2. Pretraining
3. Finetuning

## Tokenizer

Tokenization means converting the raw text as a string to some sequence of integers according to some vocabulary of possible elements

To get started I'm using the tiny_shakespeare dataset. a simple tokeniser implementation could be at the character level or word level.

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

I'm initiating the tokenizer and trying our encoder and decoder.

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

I'm using [tiktoken](https://github.com/openai/tiktoken) library which uses a byte pair encoding(BPE) algorithm for tokenization. Andrej Karpathy has a clean implementation of the BPE algorithm. 

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
```

we can encode and decode

```python
text = "we are building an agi"
ids = enc.encode(text)
print(ids)
print(enc.decode(ids))
```

```
[854, 553, 6282, 448, 1017, 72]
we are building an agi
```

How does tiktoken handles unknown words (out of vocab)? the algorithm breaks down the ou of vocab words into subword units or even at character level. See it in action [here](https://tiktokenizer.vercel.app).

Andrej has a [YouTube video](https://www.youtube.com/watch?v=zduSFxRajkE) on the BPE.

## Preparing the dataset

I'm using PyTorch classes `Dataset` and `DataLoader`. A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, text, tokenizer, block_size, stride):
        self.x = []
        self.y = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - block_size, stride):
            input = token_ids[i:i + block_size]
            target = token_ids[i + 1:i + block_size + 1]
            self.x.append(torch.tensor(input))
            self.y.append(torch.tensor(target))
```

I'm creating a PyTorch dataloader to load the data in batches.

```python
block_size = 4 # what is the maximum context length for predictions?
batch_size = 1 # how many independent sequences will we process in parallel?
stride = 1 # how many positions the input shifts across batches?

data = CustomDataset(text, tokenizer, block_size, stride)

train_dataloader = DataLoader(dataset=data, 
                              batch_size=batch_size,
                              num_workers=1,
                              shuffle=True)
```

see the input and target

```python
data_iter = iter(train_dataloader)
x, y = next(data_iter)

print("inputs:")
print(x)
print("targets:")
print(y)
```

```
inputs:
tensor([[ 10652,  70176,    412,  59509],
        [ 11062,    290,  77689,    198],
        [   889,    357,   1715,  10304],
        [  1632,  15939,  54659,   1076],
        [   484,  37510,    413,  75843],
        [ 32618,     11,    357,    198],
        [157223,    198, 197964,  12698],
        [  1661,     11,   1954,    501]])
targets:
tensor([[ 70176,    412,  59509, 113080],
        [   290,  77689,    198,   2566],
        [   357,   1715,  10304,   1262],
        [ 15939,  54659,   1076,    364],
        [ 37510,    413,  75843,    261],
        [    11,    357,    198,   4117],
        [   198, 197964,  12698,    306],
        [    11,   1954,    501,  32195]])
```

## Converting tokens to their embeddings

I'm using an inbuilt pytorch module `nn.Embedding`. This layer works like a lookup operation. For example, in the input sequence if we want to get an embedding for the token_id `5`, look into the `6`th row of input_embedding_matrix and pluck the row out. That's the embedding for token_id `5`.


```python
vocab_size = 50257
output_dim = 256 # GPT3 uses 12,288 dim
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = embedding_layer(x)
```

## Add positional encoding to encode positional information

in original paper these are fixed but since GPT2 these are like regular parameters that are learned in training

```python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # 0 1 ... max_length-1
```

token embeddings are then added to the positional embeddings to get the final input embeddings

```python
input_embeddings = token_embeddings + pos_embeddings
```

## Attention mechanism

attention mechanism is the aggregation/reduce function

self-attention mechanism allows each token to attend to tokens in all other positions and it works as follows:

step 1: calculate attention scores by taking dot product of current token (query) with all the other tokens (keys)

step 2: use softmax to normalize each attention scores (generally done for training stability)

step 3: multiply input embeddings with attention scores and then sum up all the resulting vectors

![Image](https://jalammar.github.io/images/t/self-attention-output.png)


> [!NOTE]
> query => what i am looking for
>
> key => what do i contain
>
> value => if you find me interesting heres what i am communicating to you


this is vanilla self-attention. there're further optimized versions to it. causal attention that masks the future tokens and multi-head attention that distributed causal attention to multiple heads.

## Transformer 

