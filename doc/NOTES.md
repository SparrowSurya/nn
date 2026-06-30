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

```
    ┌────────────────────────────────────────────────────────┐
    │               ARTIFICIAL INTELLIGENCE (AI)             │
    │  (Broadest category: Systems that simulate human mind) │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │               MACHINE LEARNING (ML)              │  │
    │  │  (Systems that learn patterns from data)         │  │
    │  │  ┌────────────────────────────────────────────┐  │  │
    │  │  │               DEEP LEARNING (DL)           │  │  │
    │  │  │  (Neural Networks with multiple layers)    │  │  │
    │  │  │  ┌──────────────────────────────────────┐  │  │  │
    │  │  │  │       LARGE LANGUAGE MODELS (LLMs)   │  │  │  │
    │  │  │  │  (Deep Transformers trained on text) │  │  │  │
    │  │  │  └──────────────────────────────────────┘  │  │  │
    │  │  └────────────────────────────────────────────┘  │  │
    │  └──────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────┘
```
1. Artificial Intelligence (AI): The overall umbrella. Any system that mimics human intelligence, including old rule-based systems from the 1960s (like a chess bot that uses a decision tree).
2. Machine Learning (ML): A sub-field of AI. Instead of writing hard-coded rules, we give the computer data and let it learn the rules itself (like a Spam Filter predicting spam based on word frequencies).
3. Deep Learning (DL): A sub-field of ML. Specifically refers to using Deep Neural Networks (networks with many layers, like the multi-layer network you built in main.py).
4. Large Language Models (LLMs): A specialized application of Deep Learning. They are deep neural networks utilizing the Transformer architecture, scaled up to billions of parameters, and trained specifically to understand and generate natural language.


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

### Output Layer Activation Selection Summary
Depending on the task, different activation functions are selected for the output layer to shape the output values:

| Task Type | Output Activation | Output Range | Example |
| :--- | :--- | :--- | :--- |
| **Binary Classification** | Sigmoid | `0` to `1` (probability) | Predict XOR output, spam vs. not spam |
| **Multi-class Classification** | Softmax | `0` to `1` (summing to `1.0` across all outputs) | Predict digit `0` through `9` |
| **Regression** | None (Linear) | $-\infty$ to $+\infty$ | Predict housing prices, temperatures |

---

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

### II. Leaky ReLU
A variant of ReLU designed to prevent neurons from "dying".

* **Math**:
  $$f(x) = \max(\alpha x, x) \quad (\text{typically } \alpha = 0.01)$$
* **When to Use**: Used in hidden layers when regular ReLU suffers from "dying ReLU" (where neurons get stuck outputting `0` for all inputs and stop updating).

---

### III. Sigmoid
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

### IV. Tanh (Hyperbolic Tangent)
A zero-centered activation function that squashes values between `-1` and `1`.

* **Math**:
  $$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
* **Graph**:
  ```text
     1.0 +       _--~~~
         |     _-
     0.0 +----+---- (0, 0)
         |  _-
    -1.0 +_--
  ```
* **When to Use**: Historically used in hidden layers before ReLU. It is zero-centered, meaning negative inputs map to negative outputs, making optimization easier than Sigmoid.

---

### V. Softmax
Converts a vector of numbers into a vector of probabilities that sum to `1.0`.

* **Math**:
  $$f(x_i) = \frac{e^{x_i}}{\sum_{k} e^{x_k}}$$
* **When to Use**: Used exclusively in the **output layer** for multi-class classification problems.

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

---

### II. Mean Absolute Error (MAE / L1 Loss)
Calculates the average of the absolute differences between the predictions and targets.

* **Math**:
  $$L = |y - \hat{y}|$$
* **When to Use**: Regression tasks where you want to be robust to outliers. Because errors are not squared, outliers do not dominate the gradient updates.

---

### III. Binary Cross-Entropy (BCE)
Measures the performance of a classification model whose output is a probability value between 0 and 1.

* **Math**:
  $$L = - \Big( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \Big)$$
* **When to Use**: The industry standard for binary classification problems (such as the XOR gate). It heavily penalizes incorrect predictions that are confident, which speeds up optimization.

---

### IV. Categorical Cross-Entropy
Measures performance for multi-class classification.

* **Math**:
  $$L = -\sum_{i} y_i \log(\hat{y}_i)$$
* **When to Use**: Used in multi-class classification problems, typically paired with a Softmax output layer.

---

## 5. Training and Backpropagation

The training process consists of iteratively optimizing the network's parameters (weights and biases) to minimize the loss. This is broken down into structured steps.

### I. The Modular Training Loop
To keep the network architecture understandable, training is split into four distinct steps:

1. **Forward Step**: Propagates the input through the network to generate predictions, caching the input values of each layer so they are available for backpropagation.
2. **Loss Calculation**: Evaluates how close the prediction is to the target.
3. **Backward Step**: Goes backwards through the network layers, calculating how changing each weight/bias impacts the overall loss.
4. **Parameter Updates**: Adjusts the weights and biases using **Gradient Descent**:
   $$W \leftarrow W - (\alpha \cdot \text{gradient}_W)$$
   $$b \leftarrow b - (\alpha \cdot \text{gradient}_b)$$
   *(where $\alpha$ is the learning rate)*

---

### II. Backpropagation Mathematics
Backpropagation uses the **Calculus Chain Rule** to calculate the contribution of each weight to the output error.

For a single neuron $j$ in a layer receiving inputs $x_i$:

1. **Calculate Delta ($\delta_j$)**: The error signal for neuron $j$, representing how much the pre-activation sum $z_j$ affects the loss.
   $$\delta_j = \text{error}_j \cdot \sigma'(z_j)$$
   *(where $\sigma'(z_j)$ is the derivative of the activation function at that output)*

2. **Update Bias**: Adjust the bias directly proportional to the delta:
   $$b_j \leftarrow b_j - \alpha \cdot \delta_j$$

3. **Update Weight**: Adjust the weight proportional to delta scaled by the input $x_i$ that came into it:
   $$w_{j, i} \leftarrow w_{j, i} - \alpha \cdot \delta_j \cdot x_i$$

4. **Pass Gradient Backward**: Calculate the error contribution for the previous layer to continue backpropagation:
   $$\text{next\_gradient}_i = \sum_j \delta_j \cdot w_{j, i}$$


## Deciding architecture
Here is how you decide these core architectural components when designing a neural network:

### 1. Size of Input and Output Layers

These are the easiest to decide because they are strictly determined by your data and task, not by choice.

* Input Layer Size: Matches the number of features in a single input sample.
  * Example (XOR): The inputs are pairs like  [0, 1] , so the size is  2 .
  * Example (Image): A 28 × 28 pixel image has 784 total pixels, so the input size is  784  (when flattened).
* Output Layer Size: Matches the number of values you want the network to predict.
  * Example (Binary Classification / XOR): You want a single probability (is the output  0  or  1 ?), so the output size is  1 .
  * Example (Digit Classification 0-9): You want to know which of the 10 digits it is, so the output size is  10 .


### 2. Number and Size of Hidden Layers

Choosing the number of hidden layers and their neuron count is one of the most critical aspects of neural network design. While it is partly an empirical science (trial and error), there are established mathematical theorems and practical heuristics that guide this process.

---

#### A. Number of Hidden Layers (Depth)

The depth of a network determines its level of abstraction. 

* **0 Hidden Layers**: Can only represent **linearly separable** functions. It behaves exactly like a collection of linear regression or perceptron models (e.g., AND/OR gates, linear classification).
* **1 Hidden Layer**: According to the **Universal Approximation Theorem** (Hornik, 1989), a single hidden layer with a finite number of neurons and a non-linear activation function can approximate any continuous function to arbitrary precision. 
  * *Verdict*: A single hidden layer is sufficient for simple datasets like XOR or MNIST digit classification.
* **2+ Hidden Layers (Deep Networks)**: While a single layer can theoretically represent any function, it may require an impractically large number of neurons (width). Adding layers (depth) allows the network to learn **hierarchical feature representations** parameter-efficiently.
  * *Example*: In computer vision, early layers learn edges, middle layers group edges into shapes, and deep layers group shapes into complex objects. Yoshua Bengio (2009) demonstrated that deep architectures can represent highly complex functions with exponentially fewer parameters than shallow architectures.

---

#### B. Size of Hidden Layers (Width)

Choosing the number of neurons ($N_h$) in a hidden layer is a trade-off between network capacity and generalization. 

* **Too Few Neurons**: **Underfitting**. The network lacks the capacity to capture the complex relationships in the training data.
* **Too Many Neurons**: **Overfitting**. The network will memorize the training noise instead of generalizing to unseen data, and training will be computationally slow.

##### Standard Design Heuristics (Rules of Thumb)

Let $N_i$ be the number of inputs, and $N_o$ be the number of outputs.

1. **The Geometric Mean / Pyramid Rule**:
   A classic heuristic suggests that the hidden layer size should form a smooth bottleneck between the input and output sizes:
   $$N_h = \sqrt{N_i \times N_o}$$
   * *MNIST Example*: For $784$ inputs and $10$ outputs: $\sqrt{784 \times 10} \approx 88.5$ neurons.

2. **The Interpolation Rule**:
   The hidden layer size is typically between the input size and output size, often recommended as:
   $$N_h = \frac{2}{3} N_i + N_o$$
   * *MNIST Example*: $\frac{2}{3}(784) + 10 \approx 532$ neurons.
   * Generally, $N_h$ should satisfy: $N_o < N_h < N_i$.

3. **The Sample Size Bound (Overfitting Prevention)**:
   To prevent overfitting when training without regularization (like dropout or weight decay), the number of parameters (weights + biases) should be significantly smaller than the number of training samples ($N_s$).
   For a single hidden layer, the total parameters is $P = N_h(N_i + N_o + 1)$. A common guideline is to keep $P \le \frac{N_s}{c}$ where $c \ge 2$:
   $$N_h \le \frac{N_s}{c \cdot (N_i + N_o + 1)}$$
   * *MNIST Example*: If training on $N_s = 1000$ samples: $N_h \le \frac{1000}{2 \cdot (784 + 10 + 1)} \approx 0.63$ neurons. This mathematically shows why training a wide network on very few samples requires strong regularization or using a smaller input representation (like downsampling) to prevent overfitting!

##### Practical Workflow
1. **Start Small**: Start with a single hidden layer and a small number of neurons (e.g., $32$ or $64$ for MNIST).
2. **Monitor Validation Loss**: If the network underfits (high training loss), increase the width (neurons) or depth (layers).
3. **Apply Regularization**: If the network overfits (training loss decreases, but validation loss increases), decrease the width or add regularization (like weight decay or dropout).

### 3. Which Activation Function to Use?

This depends entirely on where the layer is in the network.
#### A. In Hidden Layers (Standard Default: ReLU)

* ReLU (Rectified Linear Unit) is the industry standard for hidden layers. It is computationally very fast (just max(0,x)) and avoids the vanishing
gradient problem, allowing deep networks to train successfully.
* Sigmoid or Tanh are rarely used in hidden layers today because they "saturate" (when inputs are very high or very low, their gradients become nearly  0 ,
which halts learning).
#### B. In the Output Layer (Determined by Task Type)
The output activation shapes the final prediction format:
  Task Type                     | Output Activation            | Output Range                                     | Example
-------------------------------|------------------------------|--------------------------------------------------|---------------------------------------
  Binary Classification         | Sigmoid                      |  0  to  1  (probability)                         | Predict XOR output, spam vs. not spam
  Multi-class Classification    | Softmax                      |  0  to  1  (summing to  1.0  across all outputs) | Predict digit  0  through  9
  Regression                    | None (Linear)                | -∞ to +∞                                         | Predict housing prices, temperatures

