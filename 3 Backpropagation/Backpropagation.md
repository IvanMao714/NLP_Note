# Backpropagation

How do we train to yield great prediction? 

1. **Define a loss function $$L(\theta)$$**
2. **Calculate the gradient of $$L(\theta)$$**
3. **Take a step in the direction of the negative gradient**: Minimizing the loss:   $$\theta_{new} = \theta_{old} - n\frac{dL}{d\theta}$$![](.\img\1+.png)

## Example: Single Neuron

<img src=".\img\2+.png"  />

### STEP1: Forward propagation

1. compute the value of each neuron h, o using the model equations

2. predict the value of y: o

   **Goal: make o as close to y as possible**

### STEP2: Compute loss function

Using square loss in this example for simplicity $$L = \frac{1}{2}(y -o )^{2}$$

### STEP3: Compute gradient weight $$\theta$$

## Start with $$\frac{dL}{dw_{L}}$$

[Can read the material]: ./Backpropagation(补充).md

