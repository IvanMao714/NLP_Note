# NLM implementation

## 输入文本

```python
sentences = [
             'bob likes sheep',
             'alice is fast',
             'cs685 is fun',
             'i love lamp'
]
```

## 列举对应的编号

```python
# given the first two words of each sentence, we'll try to predict the third word using a fixed window NLM

# before we start any modeling, we have to tokenize our input and convert the words to indices

vocab = {} # map from word type to index 统计出所有不同输入的词
inputs = [] # store an indexified version of each sentence 列出input的词的编号

for sent in sentences:
  sent_idxes = []

  words = sent.split()
  for w in words:
    if w not in vocab:
      vocab[w] = len(vocab) # add a new word type
    sent_idxes.append(vocab[w])
  
  inputs.append(sent_idxes)
```

> vocab： {'bob': 0, 'likes': 1, 'sheep': 2, 'alice': 3, 'is': 4, 'fast': 5, 'cs685': 6, 'fun': 7, 'i': 8, 'love': 9, 'lamp': 10}
>
> inputs： [[0, 1, 2], [3, 4, 5], [6, 4, 7], [8, 9, 10]]

## 分出Label和prefixes（input）

```python
import torch

# two things:input
# 1. convert to LongTensor
# 2. define inputs/outputs, the first two words and the third word
prefixes = torch.LongTensor([sent[:2] for sent in inputs])
labels = torch.LongTensor([sent[2] for sent in inputs])

```

## 构建网络

```python
import torch.nn as nn

class NLM(nn.Module):
    # two things you need to do
    # 1. init function (initializes all the **params** of the network)
    # 2. forward function (defines the forward computations)
    def __init__(self, d_embedding, d_hidden, window_size, len_vocab):
        super(NLM, self).__init__() # initialize the base Module class
        self.d_embs = d_embedding 
        self.embeds = nn.Embedding(len_vocab, d_embedding)
        # concatenate embeddings > hidden
        self.W_hid = nn.Linear(d_embedding * window_size, d_hidden)
        # hidden > output probability distribution over vocab
        self.W_out = nn.Linear(d_hidden, len_vocab)



    def forward(self, input): # each input will be a batch of prefixes
        batch_size, window_size = input.size()
        embs = self.embeds(input) # 4 x 2 x 5

        # next, concatenate the prefix embeddings together
        concat_embs = embs.view(batch_size, window_size * self.d_embs) # 4 x 10
        
        # we project this to the hidden space
        hiddens = self.W_hid(concat_embs) # 4 x d_hidden
        # finally, project hiddens to vocabulary space
        outs = self.W_out(hiddens)
        

        # probs = nn.functional.softmax(outs, dim=1)

        return outs # return unnormalized probability, alsk known as "logits"
    
network = NLM(d_embedding=5, d_hidden=12, window_size=2, len_vocab=len(vocab))
network(prefixes)
```

> tensor([[-0.0430,  0.3760, -0.0404,  0.3504,  0.4570, -0.2930, -0.4358,  0.1522,          0.0560, -0.0793, -0.3191],        [-0.4502, -0.2621,  0.2912,  0.1220,  0.3365, -0.4783, -0.0571, -0.1595,          0.2648, -0.5140, -0.3851],        [-0.3595, -0.0772,  0.2197,  0.2927,  0.2890, -0.2596, -0.1193, -0.2810,         -0.0440, -0.7078, -0.5025],        [-0.0506, -0.3506,  0.4256,  0.5166,  0.5341, -0.3084,  0.2623, -0.0991,          0.0066,  0.0313, -0.0552]], grad_fn=<AddmmBackward>)

## 设置超参数

```python
num_epochs = 30
learning_rate = 0.1

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=network.parameters(), lr=learning_rate)
```

## 训练模型

```python
# training loop
for i in range(num_epochs):
    logits = network(prefixes)
    loss = loss_fn(logits, labels)
    print(f'epochs[{i+1}/{num_epochs}]loss: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

## 预测结果

```python
rev_vocab = dict((idx, word) for (word, idx) in vocab.items())
boblikes = prefixes[0].unsqueeze(0)
logits = network(boblikes)
probs = nn.functional.softmax(logits, dim=1).squeeze()

argmax_idx = torch.argmax(probs).item()
print(probs)
print(argmax_idx)
print(f'given "bob likes", the model prediction as next word  is: [{rev_vocab[argmax_idx]}], probability is {probs[argmax_idx]}')
```

> tensor([1.6756e-03, 2.5217e-03, 9.7605e-01, 1.0478e-03, 2.4076e-03, 6.0406e-03,        2.2282e-03, 4.8825e-04, 2.3747e-03, 3.0853e-03, 2.0833e-03],       grad_fn=<SqueezeBackward0>) 2 given "bob likes", the model prediction as next word  is: [sheep], probability is 0.9760469198226929