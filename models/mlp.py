import torch.nn as nn


def build_mlp(in_size: int, hidden_size: int, out_size: int, *, layer_norm: bool = True, dropout: float = 0.0):
    # Build the initial two-layer pattern: input projection -> hidden activation.
    layers = [
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
    ]
    # Optional dropout: enabled only when dropout > 0.
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Add the second hidden layer and activation to improve representation capacity.
    layers.extend(
        [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ]
    )
    # Optional dropout after the second hidden layer.
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Project to the target output dimension.
    layers.append(nn.Linear(hidden_size, out_size))
    # Optional LayerNorm: stabilizes training and normalizes output distribution.
    if layer_norm:
        layers.append(nn.LayerNorm(out_size))

    # Assemble layers in order into a callable feed-forward network.
    return nn.Sequential(*layers)
