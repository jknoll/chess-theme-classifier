import torch
from torch.utils.data import DataLoader, random_split
from dataset import ChessPuzzleDataset
from model import Model
from metrics import jaccard_similarity

# Load the test dataset
dataset = ChessPuzzleDataset('lichess_db_puzzle_test.csv')
print(f"Dataset loaded with {len(dataset)} samples")

# Split the dataset into train and test sets
random_generator = torch.Generator().manual_seed(42)
_, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
print(f"Test dataset has {len(test_dataset)} samples")

# Initialize the model with the correct number of labels
num_labels = len(dataset.all_labels)
model = Model(num_labels=num_labels)
print(f"Model initialized with {num_labels} labels")

# Import the checkpoint utilities
from checkpoint_utils import load_checkpoint

# Load the checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "checkpoint_20250403-215449_190.pth"

# Use the unified checkpoint loading function
model = model.to(device)
# Set strict=False to allow loading checkpoints between models with different architectures
checkpoint_info = load_checkpoint(checkpoint_path, model, device=device, strict=False)
model.eval()

print(f"Loaded checkpoint: {checkpoint_path}")
print(f"Checkpoint epoch: {checkpoint_info['epoch']}, global step: {checkpoint_info['global_step']}")
print(f"Note: Using non-strict loading to handle model architecture differences")

# Create a dataloader for the test set
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Evaluate the model
test_loss = 0.0
test_jaccard = 0.0
criterion = torch.nn.BCEWithLogitsLoss()
num_batches = 0

with torch.no_grad():
    for data in test_dataloader:
        inputs = data['board']
        inputs = inputs.unsqueeze(1).to(device)
        labels = data['themes'].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        # Calculate Jaccard similarity
        output_probs = torch.sigmoid(outputs)
        jaccard = jaccard_similarity(output_probs, labels, threshold=0.5)
        test_jaccard += jaccard.item()
        
        num_batches += 1

# Calculate average metrics
avg_test_loss = test_loss / num_batches
avg_test_jaccard = test_jaccard / num_batches

print(f"Test Loss: {avg_test_loss:.6f}")
print(f"Test Jaccard Similarity: {avg_test_jaccard:.6f}")
print("Evaluation complete")