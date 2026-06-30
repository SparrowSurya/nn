# LLM Training and Inference Pipelines

This document provides a comprehensive, step-by-step conceptual breakdown of how a Large Language Model (LLM) is built during the **Training Phase** and how it processes user prompts to generate responses during the **Prediction (Inference) Phase**.

---

## 1. The Training Phase (Building the Model)

The training phase is a resource-intensive, multi-month process designed to teach a neural network language syntax, facts, reasoning, and conversational behavior. It is split into four primary stages:

```text
┌─────────────────┐      ┌────────────────┐      ┌─────────────────┐      ┌───────────────┐
│ 1. Data Cleaning│ ───> │ 2. Tokenization│ ───> │ 3. Pre-Training │ ───> │ 4. Fine-Tuning│
└─────────────────┘      └────────────────┘      └─────────────────┘      └───────────────┘
```

### I. Data Collection and Curation
An LLM requires a massive corpus of text to learn from. This includes books, articles, code repositories, and web crawls.
* **Filtering**: Algorithms scan the raw data to remove spam, machine-generated gibberish, and formatting markup (like HTML tags).
* **Deduplication**: Duplicate articles or repetitive pages are deleted to ensure the model doesn't overfit to repeated text.
* **Safety & Privacy**: Personal Identifiable Information (PII) like phone numbers, addresses, and private keys are scrubbed.

### II. Tokenizer Training
Before text is fed into a neural network, it must be represented numerically. 
* **Vocabulary Creation**: An algorithm (typically Byte-Pair Encoding or WordPiece) scans the curated dataset to construct a dictionary of the most common characters, sub-words, and whole words (typically ranging from $32,000$ to $256,000$ unique entries).
* **Token IDs**: Each item in the vocabulary is assigned a unique integer index. For example:
  * `"hello"` $\rightarrow$ `31349`
  * `" world"` $\rightarrow$ `995`

### III. Pre-Training (Next-Token Prediction)
This is the most computationally expensive stage, taking weeks or months on clusters of thousands of GPUs. The model learns by playing a continuous game of "guess the next word" on internet text.

1. **Embedding Lookup**: The model converts incoming token IDs into high-dimensional vectors (often $4,096$ or $8,192$ dimensions) using a lookup matrix.
2. **Forward Pass**: The vectors pass through dozens of Transformer layers (containing attention mechanisms and feedforward networks) to compute representation scores.
3. **Logits & Softmax**: The final representation is mapped back to the vocabulary size, yielding a raw score (logit) for every possible word in the dictionary. A Softmax function converts these scores into a probability distribution.
4. **Loss Calculation**: The model calculates the difference between its predicted probabilities and the actual next word in the text using a loss function (Categorical Cross-Entropy).
5. **Backpropagation**: The loss error is sent backward through the network to calculate the gradient of the error with respect to every weight and bias.
6. **Gradient Descent**: An optimizer (like AdamW) updates the weights slightly to make the next prediction more accurate.

### IV. Instruction Fine-Tuning & Alignment (Post-Training)
A pre-trained base model is only good at document completion. If you prompt it with *"Write a story about a dragon,"* it might respond by outputting a list of other prompts. Post-training refines it into a conversational assistant.

* **Supervised Fine-Tuning (SFT)**: The model is trained on thousands of high-quality, human-curated question-and-answer pairs. It learns the format of instructions and responses.
* **Reinforcement Learning from Human Feedback (RLHF)**: The model generates multiple candidate answers. Humans (or grading models) rate the quality of the outputs. The model's weights are optimized (using techniques like DPO or PPO) to favor outputs that are helpful, harmless, and honest.

### V. Serialization
The completed weights and biases are saved as a serialized dictionary file (such as `.safetensors` or `.bin`) to be loaded by inference servers.

---

## 2. The Prediction Phase (Inference)

The prediction phase is the live execution loop that runs when you send a prompt to an aligned model. It is designed to be low-latency and responsive.

```text
User Input ──> Formatting ──> Tokenize ──> Embedding Lookup ──> Positional Encoding
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
│
└──> Transformer Layers (Self-Attention & MLP) ──> Logits ──> Softmax
                                                                │
┌───────────────────────────────────────────────────────────────┘
│
└──> Sampling ──> Stop Check? ──[Yes]──> End
                       │
                     [No]
                       │
                       └──> Detokenize ──> Stream to Screen
                                 │
                      (Append Token to Input)
                                 │
                                 └── (Loop to Embedding Lookup)
```

### I. Capture and Formatting
When you type a prompt, the system wraps it in a structured schema (such as ChatML tags) to clearly separate user instructions, system guidelines, and model replies:
```text
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

### II. Tokenization
The tokenizer translates the formatted string into a sequence of integer token IDs:
* **Text**: `"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"`
* **Token IDs**: `[100264, 882, 310, 220, 10, 220, 100265, 100266]`

### III. Embedding Lookup and Positional Encoding
* **Embedding Lookup**: The sequence of token IDs is passed into the input embedding matrix. Each ID retrieves its corresponding vector (e.g., $4,096$ float values).
* **Positional Encoding**: Because Transformers process all tokens simultaneously, they have no built-in concept of word order. The system adds a unique positional vector (a sequence of coordinates representing the index of each word) to the embedding vector.

### IV. Forward Pass through Transformer Blocks
The position-encoded vectors are passed through the hidden layers:
* **Self-Attention**: The vectors query one another to determine relationships. For example, in the sentence *"The bank of the river"*, the attention mechanism connects *"bank"* heavily to *"river"* to determine its meaning, rather than connecting it to financial terms.
* **Feedforward Network (MLP)**: The vectors are passed through fully connected layers and non-linear activation functions (like GELU) to build complex semantic representations.

### V. Logits Generation and Softmax
The output vector of the **last token in the sequence** is mapped back to the vocabulary space.
* **Logits**: The output is a list of raw scores (logits) of length equal to the vocabulary size (e.g., $100,000$).
* **Softmax**: The logits are run through a Softmax function, converting them into probabilities between $0.0$ and $1.0$ that sum up to $1.0$.

### VI. Sampling
The system selects a single token ID from the probability distribution. The selection is adjusted by parameters:
* **Temperature**: Controls randomness. A lower temperature (e.g., $0.2$) makes the output highly predictable by prioritizing the absolute highest probability token. A higher temperature (e.g., $0.8$) introduces variety by allowing lower probability tokens to be selected.
* **Top-P (Nucleus Sampling)**: Restricts selection to a subset of tokens whose cumulative probability exceeds a threshold $P$ (e.g., $0.90$).

### VII. Stop Check and Detokenization
* **Stop Check**: The system checks if the selected token is a special End-of-Text (`<|im_end|>`) token. If it is, the loop terminates.
* **Detokenization**: If the token is regular text, it is converted from its integer ID back into a human-readable text string.
* **Streaming**: The text string is immediately sent to the user interface and rendered on the screen.

### VIII. Autoregressive Looping
The predicted token is appended to the end of the input sequence. The network loops back to **Step III** and runs a new forward pass to predict the next token. This loop repeats until the end token is generated or the maximum context length is reached.

---

## 3. Concrete Step-by-Step Example

Let's trace the prompt `"What is 2+2?"` through a model with a vocabulary size of $100,000$:

1. **User Types**: `"What is 2+2?"`
2. **Formatting**:
   `"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"`
3. **Tokenization**:
   The string is mapped to `[100264, 882, 310, 220, 10, 220, 100265, 100266]` (8 tokens).
4. **Embeddings**:
   The 8 tokens are converted into a $8 \times 4096$ matrix of floats.
5. **Computation**:
   The matrix passes through 32 Transformer layers. The attention mechanism processes relationships between all 8 tokens.
6. **Output Prediction (First Iteration)**:
   The output vector of the 8th token passes through the final linear layer. 
   * **Logits**: The raw score for token representing `"4"` is very high; others are low.
   * **Softmax**: The probability for token representing `"4"` is computed as `0.99`.
   * **Sampling**: Token representing `"4"` (ID `15`) is selected.
7. **Streaming**:
   ID `15` is detokenized to `"4"` and displayed on your screen.
8. **Loop**:
   The new sequence is `[100264, 882, 310, 220, 10, 220, 100265, 100266, 15]` (9 tokens).
9. **Second Iteration**:
   The 9 tokens are processed. The model predicts the probability of the next token. Since the answer is complete, the highest probability token is the End-of-Text token (`100265`).
10. **Termination**:
   The system samples ID `100265`, recognizes it as `<|im_end|>`, stops the generation loop, and closes the response window.
