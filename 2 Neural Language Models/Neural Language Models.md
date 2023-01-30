# Neural Language Models

## one-hot vectors

<img src=".\img\1.png" style="zoom:75%;" />

#### Shortcoming:

**all words are equally (dis)similar:** dot product is zero! these vectors are orthogonal

## Neural networks

**embeddings:** represent words with low-dimensional vectors

<img src=".\img\2.png" style="zoom:60%;" />

neural networks compose word embeddings into vectors for phrases, sentences, and documents

<img src=".\img\3.png" style="zoom:75%;" />

**Softmax layer:** convert a vector representation into a probability distribution over the entire vocabulary

<img src=".\img\4.png" style="zoom:60%;" />

Each row of W contains feature weights for a corresponding word in the vocabulary.

Each dimension of x corresponds to a feature of the prefix

<img src=".\img\5.png" style="zoom:65%;" />

<img src="C:\Users\65151\Desktop\NLP\2 Neural Language Models\img\6.png" style="zoom:50%;" />

## Composition functions

### A fixed-window neural Language Model

<img src=".\img\7.png" style="zoom:75%;" />

<img src="C:\Users\65151\Desktop\NLP\2 Neural Language Models\img\8.png" style="zoom:75%;" />

how does this compare to a normal n-gram model?

#### Improvements over n-gram LM:

- No sparsity problem
- Model size is O(n) not O(exp(n))

#### Remaining problems:

- Fixed window is too small
- Enlarging window enlarges
- Window can never be large enough!
- Each ci uses different rows of W. We don’t share weights across the window.

### Recurrent Neural Networks

<img src=".\img\9.png" style="zoom:65%;" />

#### RNN Advantages:

- Can process any length input
- Model size doesn’t increase for longer input
- Computation for step t can (in theory) use information from many steps back
- Weights are shared across timestep -> representations are shared

#### RNN Disadvantages:

- Recurrent computation is slow
- In practice, difficult to access information from many steps back