import torch
import torch.nn as nn
import os
import yaml
import argparse
from torch.utils.data import DataLoader
from dataset import ChessPuzzleDataset
from model import Model, Lamb, get_lr_with_warmup

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test the chess theme classifier model')
parser.add_argument('--csv_file', type=str, default=os.path.join('dataset', 'lichess_db_puzzle_test.csv'),
                   help='Path to the CSV file to use (default: dataset/lichess_db_puzzle_test.csv)')
parser.add_argument('--num_samples', type=int, default=None,
                   help='Number of samples to use (default: all)')
args = parser.parse_args()

print("\n" + "="*80)
print("CHESS THEME CLASSIFIER - MODEL TEST")
print("="*80)

# Create a small dataset
print("\nInitializing dataset...")
print(f"Using dataset: {args.csv_file}")
dataset = ChessPuzzleDataset(args.csv_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get number of labels
num_labels = len(dataset.all_labels)
print(f"Dataset loaded with {len(dataset):,} samples and {num_labels} theme labels")

# Initialize model with default configuration from chess-hackathon
default_model_config = {
    "num_labels": num_labels,
    "nlayers": 2,
    "embed_dim": 64,
    "inner_dim": 320,
    "attention_dim": 64,
    "use_1x1conv": True,
    "dropout": 0.5
}

# Check if model_config.yaml exists and load it if present
model_config = default_model_config.copy()
config_path = 'model_config.yaml'
if os.path.exists(config_path):
    print(f"Found {config_path}, loading configuration...")
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
        # Update the default config with values from yaml
        for key, value in yaml_config.items():
            if key in model_config:
                model_config[key] = value
    print("Configuration loaded from YAML file")

# Always ensure num_labels is set correctly
model_config["num_labels"] = num_labels

# Print model configuration
print("\nInitializing model with configuration:")
for key, value in model_config.items():
    print(f"  {key}: {value}")

# Initialize model
model = Model(**model_config)

# Initialize optimizer with settings from chess-hackathon
lr = 0.001
weight_decay = 0.01
warmup_steps = 1000
print(f"\nInitializing Lamb optimizer with lr={lr}, weight_decay={weight_decay}, warmup_steps={warmup_steps}")
optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay)

# Print learning rate schedule examples
print("\nLearning rate schedule:")
for step in [0, warmup_steps//4, warmup_steps//2, warmup_steps, warmup_steps*2]:
    lr_factor = min(step / warmup_steps, 1.0)
    current_lr = lr_factor * lr
    print(f"  Step {step:5d}: {current_lr:.6f}")

# Get one batch for testing
data = next(iter(dataloader))
inputs = data['board']
inputs = inputs.unsqueeze(1)  # Add channel dimension
labels = data['themes']

print(f"Input shape: {inputs.shape}")
print(f"Label shape: {labels.shape}")

# Test forward pass with debug output
outputs = model(inputs, debug=True)
print(f"Output shape: {outputs.shape}")

# Test loss calculation
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(outputs, labels)
print(f"Loss: {loss.item()}")

print("Model test completed successfully!")