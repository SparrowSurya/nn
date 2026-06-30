"""
This module contains helper functions to visualize the neural network architecture,
including layers, nodes, weights, and biases.
"""

import matplotlib.pyplot as plt
from lib.network import NeuralNetwork


def visualize_network_structure(nn: NeuralNetwork):
    """Draws a 2D node-link diagram of the network's weights and biases."""
    # Extract layer dimensions (e.g., [2, 3, 1])
    layer_sizes = [nn.layers[0].input_size] + [layer.output_size for layer in nn.layers]
    num_layers = len(layer_sizes)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # 1. Compute 2D grid coordinates for every neuron
    layer_coords = []
    max_nodes = max(layer_sizes)
    for layer_idx, size in enumerate(layer_sizes):
        x = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5
        # Vertically center the nodes
        y_coords = [(i - (size - 1) / 2) / (max_nodes - 1) if max_nodes > 1 else 0.0 for i in range(size)]
        layer_coords.append((x, y_coords))

    # 2. Draw connections (Weights) first so they sit behind nodes
    for layer_idx in range(len(nn.layers)):
        layer = nn.layers[layer_idx]
        x_prev, y_prevs = layer_coords[layer_idx]
        x_curr, y_currs = layer_coords[layer_idx + 1]

        # weights shape: (output_size, input_size)
        for j in range(layer.output_size):
            for i in range(layer.input_size):
                w = layer.weights[j][i]
                # Blue for positive weights, Crimson red for negative weights
                color = 'royalblue' if w >= 0 else 'crimson'
                # Scale line thickness with weight magnitude
                linewidth = min(abs(w) * 3, 5.0)
                ax.plot([x_prev, x_curr], [y_prevs[i], y_currs[j]],
                        color=color, linewidth=linewidth, alpha=0.6, zorder=1)

                # Display the weight value shifted along the line to prevent overlapping at intersection points
                t = 0.35 if (i + j) % 2 == 0 else 0.65
                x_pos = (1 - t) * x_prev + t * x_curr
                y_pos = (1 - t) * y_prevs[i] + t * y_currs[j]
                ax.text(x_pos, y_pos, f"{w:.2f}", fontsize=8, ha='center', va='center', zorder=2,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="lightgray", lw=0.5, alpha=0.9))

    # 3. Draw neurons (Nodes) and write Bias values
    for layer_idx, (x, y_coords) in enumerate(layer_coords):
        is_input = (layer_idx == 0)

        for idx, y in enumerate(y_coords):
            if is_input:
                node_color = '#ECEFF1'  # Light gray for inputs
                label = f"Input {idx+1}"
            else:
                bias_val = nn.layers[layer_idx - 1].bias[idx]
                # Golden yellow for positive biases, Orange for negative biases
                node_color = '#FFD54F' if bias_val >= 0 else '#FF8A65'
                label = f"b: {bias_val:.2f}"

            # Draw the node circle
            circle = plt.Circle((x, y), radius=0.04, color=node_color, ec="black", lw=1.5, zorder=3) # pyright: ignore[reportPrivateImportUsage]
            ax.add_artist(circle)

            # Place label text below/next to the node
            ax.text(x, y - 0.08, label, fontsize=9, ha='center', va='center', zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5, alpha=0.8))

    # Set plot boundaries
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.6, 0.6)
    plt.title("Neural Network Weights & Biases", fontsize=14, pad=20)
    plt.show()
