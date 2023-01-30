# Attention mechanisms

## Cons of RNN

**RNNs suffer from a bottleneck problem: **The current hidden representation must encode all of the information about the text observed so far

This becomes difficult especially with longer sequences (a vector to present whole sentence)

<img src=".\img\1.png" style="zoom:75%;" />

> “you can’t cram the meaning of a whole %&@#&ing sentence into a single $*(&@ing vector!”
>
> ​																																							— Ray Mooney (NLP professor at UT Austin)

what if we use multiple vectors?

<img src=".\img\2.png" style="zoom:75%;" />

## Attention

- Attention mechanisms (Bahdanau et al.,2015）allow language models to focus on a particular part of the observed context at each time step
- Originally developed for machine translation, and intuitively similar to word alignments between different languages

In general, we have a single query vector and multiple key vectors. We want to score each query-key pair

<img src=".\img\3.png" style="zoom:75%;" />

<img src=".\img\4.png" style="zoom:75%;" />

- Attention solves the bottleneck problem
  - Attention allows decoder to look directly at source; bypass bottleneck

-  Attention helps with vanishing gradient problem
  - Provides shortcut to faraway states

- Attention provides some interpretability
  - By inspecting attention distribution, we can see
  - what the decoder was focusing on
  - We get alignment for free!
  - This is cool because we never explicitly trained an alignment system
  - The network just learned alignment by itself

### Many variants of attention

<img src=".\img\5.png" style="zoom:75%;" />

## Self-attention

can completely replace recurrence!

Each element in the sentence attends to the other elements

<img src=".\img\6.png" style="zoom:75%;" />

$$Q = W^{q}\cdot P  $$        $$K = W^{k}\cdot P $$

1. $$A = Q \cdot K$$  计算出相关性
2. $$M = A(P)V$$  计算出最后的embedding   如果有位置掩码P，则乘其值

## Multi-head self-attention

<img src=".\img\7.png" style="zoom:75%;" />

寻求不同的相关性