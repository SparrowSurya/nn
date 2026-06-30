# Transformer and Attention Study Plan

This study plan is designed to guide you step-by-step from basic neural networks to understanding the inner workings of the Transformer architecture. Use the checkboxes below to track your progress as you learn.

---

## Chapter 1: Sequential Data and Recurrent Neural Networks (RNNs)
*Understand how neural networks process sequence data (like sentences) where order matters.*

### Core Concepts to Learn
* [ ] **Sequential vs. Static Data**: Why standard feed-forward networks struggle when input sizes vary and order matters.
* [ ] **Hidden State ($h_t$)**: The network's "memory" that gets updated at each time step:
  $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
* [ ] **BPTT (Backpropagation Through Time)**: How gradients are calculated backwards through the sequence timeline.

### Limits & Bottlenecks
* [ ] **Vanishing/Exploding Gradients**: Why gradients shrink or grow exponentially over long text sequences.
* [ ] **Sequential Bottleneck**: Why RNN training cannot be parallelized across GPUs (time-step $t$ requires calculations from time-step $t-1$).

---

## Chapter 2: Long Short-Term Memory (LSTM) Networks
*Learn how gating mechanisms allow networks to maintain long-term memory over larger sequences.*

### Core Concepts to Learn
* [ ] **The Cell State ($C_t$)**: The network's "conveyor belt" memory that bypasses heavy weight multiplications.
* [ ] **The Gates**:
  * **Forget Gate ($f_t$)**: Decides what information to discard from the cell state.
  * **Input Gate ($i_t$)**: Decides what new information to store.
  * **Output Gate ($o_t$)**: Decides what part of the cell state to output as the hidden state.
* [ ] **GRU (Gated Recurrent Unit)**: A simplified version of LSTM that merges the cell state and hidden state.

---

## Chapter 3: Encoder-Decoder Architectures (Seq2Seq)
*Learn the standard model layout used for sequence-to-sequence translation tasks.*

### Core Concepts to Learn
* [ ] **The Encoder**: Processes the input sequence (e.g., English sentence) and summarizes it into a single context vector.
* [ ] **The Context Vector**: The mathematical bottleneck representation of the input.
* [ ] **The Decoder**: Takes the context vector and generates the output sequence (e.g., French translation) token-by-token.

### Limits & Bottlenecks
* [ ] **Information Bottleneck**: The impossibility of compressing a long paragraph into a single fixed-length vector without losing crucial details.

---

## Chapter 4: The Attention Mechanism
*Learn how allowing the model to look back at the entire input sequence solves the information bottleneck.*

### Core Concepts to Learn
* [ ] **Alignment Scores**: Calculating similarity between the decoder's current state and all encoder hidden states.
* [ ] **Attention Weights**: Using a Softmax function to convert alignment scores into a probability distribution.
* [ ] **Context Vector as a Weighted Sum**: Multiplying the attention weights by the encoder states to create a customized vector for each step.

---

## Chapter 5: Self-Attention (The Transformer Core)
*Learn how a single sentence calculates context internally by comparing every word with every other word.*

### Core Concepts to Learn
* [ ] **The Query, Key, and Value ($Q, K, V$) Vectors**:
  * **Query ($Q$)**: "What am I looking for?"
  * **Key ($K$)**: "What information do I contain?"
  * **Value ($V$)**: "What content do I output if selected?"
* [ ] **Scaled Dot-Product Attention Equation**:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* [ ] **Scaling Factor ($\sqrt{d_k}$)**: Why dividing by the square root of the key dimension is necessary to prevent vanishing gradients in the Softmax function.

---

## Chapter 6: Multi-Head Attention
*Understand how splitting self-attention into parallel channels allows the model to focus on multiple relationships simultaneously.*

### Core Concepts to Learn
* [ ] **Projection Matrices**: Projecting $Q, K, V$ into smaller subspaces.
* [ ] **Parallel Heads**: Running multiple attention calculations (e.g., 8 heads) at the same time.
* [ ] **Concatenation**: Merging the outputs of all attention heads back into a single vector.

---

## Chapter 7: The Complete Transformer Block
*Assemble the attention mechanisms, normalizations, and feedforward networks into the final Transformer architecture.*

### Core Concepts to Learn
* [ ] **Positional Encodings**: Adding coordinates (often sine and cosine wave values) to the static embeddings so the model knows word positions.
* [ ] **Residual (Skip) Connections**: Bypassing layers to allow gradients to flow directly, enabling deep networks.
* [ ] **Layer Normalization**: Stabilizing the mean and variance of activations.
* [ ] **The Feed-Forward Layer**: The fully connected MLP block (similar to basic neural networks) that processes each token individually after attention mixing.
