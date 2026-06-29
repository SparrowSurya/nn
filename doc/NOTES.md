# Neural Network Study Notes

This document captures the current learning progress and conceptual understanding of building a Neural Network from scratch in Python.

---

## 1. What is a Neural Network?

A **Neural Network** (or Artificial Neural Network) is a computational model inspired by the structure and function of biological brains. It is designed to recognize patterns, approximate complex mathematical functions, and solve problems like classification, regression, and pattern generation.

At its core, a neural network is a **universal function approximator**. Given an input $X$, it maps it to an output $Y$ through a series of weighted calculations and non-linear transformations.

### Real-World Example
Imagine predicting whether a house will sell for a high price based on two inputs: **Size (sq ft)** and **Number of Bedrooms**.
* The network takes these inputs, weights them according to their importance, adds a bias, and outputs a prediction (e.g., probability of a high sale price).
* Through training, the network adjusts the weights (e.g., realizing that Size is more important than the number of bedrooms) to minimize prediction error.

### Pros and Cons

| Pros | Cons |
| :--- | :--- |
| **Highly Flexible**: Can model extremely complex, non-linear relationships. | **Data Hungry**: Requires a large amount of training data to perform well. |
| **No Manual Feature Engineering**: Automatically learns features from raw data. | **Black Box**: Hard to interpret *why* a network made a specific prediction. |
| **Generalization**: Performs well on unseen data if trained correctly. | **Computationally Expensive**: Requires significant CPU/GPU resources to train. |

---

## 2. Structure of a Neural Network

A neural network is organized into layers of interconnected nodes (neurons).

### ASCII Structure Diagram

```text
Input Layer            Hidden Layer           Output Layer
 (2 Inputs)             (3 Neurons)            (2 Outputs)

   (x1)  -------------\ /--- (h1) -------------\
           \          X             \           \--- (y1)
            \        / \             \         /
             \      /   \             \       /
              \    /     \             \     /
               \  /       \--- (h2) ----\---/
                \/       /             / \ /
                /\      /             /   X
               /  \    /             /   / \
              /    \  /             /   /   \--- (y2)
   (x2)  ------------/ \---- (h3) -------------/

Weights: W1 (3x2 matrix)          Weights: W2 (2x3 matrix)
Biases:  b1 (vector of 3)          Biases:  b2 (vector of 2)
```

### Explanation of Key Terms

* **Input Layer**: Receives the raw features from the dataset. No computation happens here.
* **Hidden Layer(s)**: Intermediate layers that extract abstract features from the input. Each neuron in a hidden layer receives inputs from all neurons in the previous layer.
* **Output Layer**: Produces the final prediction of the network.
* **Neuron (Node)**: The basic unit of computation. It computes the weighted sum of its inputs, adds a bias, and passes the result through an activation function.
* **Weights ($W$)**: Parameters representing the strength of connection between neurons. They determine how much influence an input has on the output of the next neuron.
* **Biases ($b$)**: An offset parameter added to the weighted sum. It allows the activation function to shift left or right, enabling the neuron to fire even when all inputs are zero.

---

## 3. Activation Functions

Activation functions introduce **non-linearity** into the network. Without them, a neural network—no matter how many layers it has—would behave like a single linear regression model ($y = wx + b$), failing to learn non-linear patterns (like the XOR gate).

### I. ReLU (Rectified Linear Unit)
The most widely used activation function for hidden layers in modern deep learning.

* **Math**:
  $$f(x) = \max(0, x)$$
* **Graph**:
  ```text
    y
    |       /
    |      /
    |     /
    |    /
    |   /
  --+-------- x
    |
  ```
* **When to Use**: Almost always used in hidden layers. It is computationally efficient and helps prevent the **vanishing gradient problem** (where gradients become too small during backpropagation, stopping the network from learning).

---

### II. Sigmoid
A classic activation function that squashes any input value into a range between `0` and `1`.

* **Math**:
  $$f(x) = \frac{1}{1 + e^{-x}}$$
* **Graph**:
  ```text
    1.0 +         _--~~~
        |       _-
    0.5 +      /  <-- Passes through (0, 0.5)
        |    _-
    0.0 + _--___________
         -3 -2 -1  0  1  2  3
  ```
* **When to Use**: Primarily used in the **output layer** for binary classification tasks, as its output can be interpreted directly as a probability. It is rarely used in hidden layers because it suffers from the vanishing gradient problem when inputs are very large or very small.

---

## 4. Loss Functions

A **Loss Function** measures how far off the network's predictions are from the true targets. It serves as the primary metric that the network minimizes during training.

### I. Mean Squared Error (MSE)
Calculates the average of the squared differences between the predicted outputs ($\hat{y}$) and true targets ($y$).

* **Math**:
  $$L = \frac{1}{2} (y - \hat{y})^2$$
* **When to Use**: Simplest and most intuitive function, often used in regression tasks or simple demonstration networks.
* **Derivative**:
  $$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$
  This clean derivative represents the direct error signal used to correct the weights during backpropagation.

### II. Binary Cross-Entropy (BCE)
Measures the performance of a classification model whose output is a probability value between 0 and 1.

* **Math**:
  $$L = - \Big( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \Big)$$
* **When to Use**: The industry standard for binary classification problems (such as the XOR gate). It heavily penalizes incorrect predictions that are confident (e.g., predicting `0.01` when the target is `1`), which speeds up optimization.
