# Basic training loop insired by https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# This script trains locally on a single GPU.
# This version has a correction to ensure that it is training on the GPU; previously I was not correctly calling model.to(device), inputs.to(device), and labels.to(device).

from cycling_utils import TimestampedTimer
import time
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

# Create logs and checkpoints directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

timer = TimestampedTimer("Imported TimestampedTimer")

from dataset import ChessPuzzleDataset
import torch
from torch.utils.data import DataLoader, random_split
from model import Model

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def jaccard_similarity(tensor1, tensor2):
    """
    Compute Jaccard similarity between two binary tensors.
    
    Args:
        tensor1: Binary PyTorch tensor
        tensor2: Binary PyTorch tensor of the same shape as tensor1
    
    Returns:
        Jaccard similarity score (float between 0 and 1)
    """
    intersection = torch.logical_and(tensor1, tensor2).sum()
    union = torch.logical_or(tensor1, tensor2).sum()
    
    return intersection.float() / (union.float() + 1e-8)  # add small epsilon to avoid division by zero

def main():
    # Set up the device for GPU use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.autograd.set_detect_anomaly(True)

    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', f'run_{timestamp}')
    writer = SummaryWriter(log_dir)
    timer.report(f"Created TensorBoard writer at {log_dir}")

    dataset = ChessPuzzleDataset('lichess_db_puzzle.csv')
    
    # Get the number of labels from the dataset
    num_labels = len(dataset.get_theme_names())
    print(f"Number of unique labels (themes + opening tags): {num_labels}")
    
    # Create model with the correct number of labels
    model = Model(num_labels=num_labels)
    model = model.to(device)

    # Get a single item
    sample = dataset[0]
    print(f"Example dataset item: {sample}")
 #   print(f"Theme names: {dataset.get_theme_names()}")
 #   print(f"Theme count: {len(dataset.get_theme_names())}")

    input = sample['board'].unsqueeze(0).unsqueeze(0).to(device) # More elegant way to do this?

    print(model)

    print(input)
    out = model(input)
    print(f"Out: {out}")

    target = sample['themes'].unsqueeze(0).to(device)
    print(f"Target: {target}")
    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss = criterion(out, target)
    print(f"Loss: {loss}")

    ############################################################
    # Example of backprop after single forward pass


    model.zero_grad()     # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(model.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(model.conv1.bias.grad)

    ############################################################

    # Split the dataset into train and test sets.    
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
    timer.report(f"Intitialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test board evaluations.")


    # Use standard samplers with DataLoader for DataParallel
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    timer.report("Prepared dataloaders")

    # Track global step for TensorBoard
    global_step = 0

    for epoch in range(10):  # loop over the dataset multiple times

        for i, data in enumerate(train_dataloader, 0):
            running_loss = 0.0
            running_jaccard_index = 0.0
            
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['board']  # Shape: [batch_size, height, width]
            inputs = inputs.unsqueeze(1).to(device)  # Add channel dimension: [batch_size, channels=1, height, width]
            labels = data['themes'].to(device)  # Remove the unsqueeze since batch dimension is already there

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Apply sigmoid to get probabilities for Jaccard calculation
            output_probs = torch.sigmoid(outputs)
            jaccard_loss = jaccard_similarity(output_probs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            running_jaccard_index = jaccard_loss.item() 
            print(f"epoch: {epoch} loss: {running_loss:.8f} jaccard index: {running_jaccard_index:.8f} steps: {i+1}")
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', running_loss, global_step)
            writer.add_scalar('Jaccard/train', running_jaccard_index, global_step)
            global_step += 1
            
            # Save checkpoint
            if i % 100000 == 0:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                checkpoint_filename = f"checkpoints/checkpoint_{timestamp}_{i}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'global_step': global_step,
                    'jaccard_loss': jaccard_loss,
                }, checkpoint_filename)
                
                # Also save a copy as the latest checkpoint for easy resuming
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'global_step': global_step,
                    'jaccard_loss': jaccard_loss,
                }, "checkpoints/checkpoint_resume.pth")
                
                # Log model checkpoint to TensorBoard
                writer.add_text('Checkpoint', f'Saved checkpoint: {checkpoint_filename}', global_step)
    
    # Close the TensorBoard writer
    writer.close()
    timer.report("Closed TensorBoard writer")

if __name__ == "__main__":
    main()