# Transfer learning with neural language models

## What is transfer learning

In our context: take a network trained on a task for which it is easy to generate labels, and adapt it to a different task for which it is harder.

- In computer vision: train a CNN on ImageNet, transfer its representations to every other CV task
- In NLP: train a really big language model on billions of words, transfer to every NLP task!

<img src=".\img\1.png" style="zoom:75%;" />

### Contextual Representations

<img src=".\img\2.png" style="zoom:75%;" />

#### History of Contextual Representations

<img src=".\img\3.png" style="zoom:75%;" />

<img src=".\img\4.png" style="zoom:75%;" />

ELMo representations are contextual: they depend on the entire sentence in which a word is used.

## BERT

### Problem with Previous Methods

Language models only use left context or right context, but language understanding is bidirectional

<img src="C:\Users\65151\Desktop\NLP\7 Transfer learning with neural language models\img\5.png" style="zoom:75%;" />

### Masked LM



### Model Architecture

<img src="C:\Users\65151\Desktop\NLP\7 Transfer learning with neural language models\img\6.png" style="zoom:75%;" />

<img src="C:\Users\65151\Desktop\NLP\7 Transfer learning with neural language models\img\7.png" style="zoom:75%;" />

### Fine-Tuning Procedure

<img src="C:\Users\65151\Desktop\NLP\7 Transfer learning with neural language models\img\8.png" style="zoom:75%;" />

### Develop History

<img src="C:\Users\65151\Desktop\NLP\7 Transfer learning with neural language models\img\9.png" style="zoom:75%;" />