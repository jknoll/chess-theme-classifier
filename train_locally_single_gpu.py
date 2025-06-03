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
from class_weights import compute_label_weights
from metrics import jaccard_similarity as metrics_jaccard_similarity
from metrics import precision_recall_f1, get_classification_report

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# No longer need this local jaccard_similarity function since we're using the one from metrics.py

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

    # Use a smaller dataset for testing with class-conditional augmentation by default
    print("Using smaller dataset with class-conditional augmentation for testing...")
    dataset = ChessPuzzleDataset('lichess_db_puzzle_test.csv', class_conditional_augmentation=True)
    
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

    # Need to match shapes: out is [batch_size, num_labels], target should be the same
    target = sample['themes'].unsqueeze(0).to(device)
    out = out.unsqueeze(0) if out.dim() == 1 else out  # Ensure out has batch dimension
    print(f"Target: {target}")
    # Define the loss function with class weights for cost-sensitive learning
    print("Computing class weights for weighted BCE loss...")
    # Compute label weights
    pos_weights = compute_label_weights(dataset)
    pos_weights = pos_weights.to(device)
    print("Using weighted BCE loss for class imbalance mitigation")
    
    # Use weighted BCE loss by default
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Use Lamb optimizer from the main training script for consistency
    from model import Lamb
    lr = 0.001
    weight_decay = 0.01
    optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss = criterion(out, target)
    print(f"Loss: {loss}")

    ############################################################
    # Example of backprop after single forward pass


    model.zero_grad()     # zeroes the gradient buffers of all parameters

    print('First layer params grad before backward')
    first_layer = model.convLayers[0].conv1
    print(first_layer.bias.grad)

    loss.backward()

    print('First layer params grad after backward')
    print(first_layer.bias.grad)

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

    for epoch in range(2):  # loop over the dataset a few times for testing

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
            
            # Debug shape issues on first batch
            if i == 0:
                print(f"Model output shape: {outputs.shape}")
                print(f"Labels shape: {labels.shape}")
            
            # Ensure outputs and labels have compatible shapes
            if outputs.dim() == 1 and labels.dim() == 2:
                # If output is a 1D tensor but labels are 2D (batch, features)
                outputs = outputs.unsqueeze(0)  # Add batch dimension
            elif outputs.dim() == 2 and labels.dim() == 2:
                # Both are 2D, check if batch sizes match
                if outputs.size(0) != labels.size(0):
                    # Reshape outputs to match batch size if needed
                    outputs = outputs.view(labels.size(0), -1)
            
            # Double-check shapes on first batch
            if i == 0:
                print(f"Adjusted output shape: {outputs.shape}")
                
            loss = criterion(outputs, labels)
            # Apply sigmoid to get probabilities for metrics calculation
            output_probs = torch.sigmoid(outputs)
            jaccard_loss = metrics_jaccard_similarity(output_probs, labels, threshold=0.5)
            
            # Calculate precision, recall, F1 metrics periodically
            calculate_detailed_metrics = (i % 10 == 0)  # More frequent for single-GPU version

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            running_jaccard_index = jaccard_loss.item() 
            print(f"epoch: {epoch} loss: {running_loss:.8f} jaccard index: {running_jaccard_index:.8f} steps: {i+1}")
            
            # Calculate and log additional metrics periodically
            metrics_dict = {}
            if calculate_detailed_metrics:
                # Calculate precision, recall, F1 with different averaging methods
                precision_micro, recall_micro, f1_micro = precision_recall_f1(
                    output_probs, labels, threshold=0.5, average='micro'
                )
                precision_macro, recall_macro, f1_macro = precision_recall_f1(
                    output_probs, labels, threshold=0.5, average='macro'
                )
                
                metrics_dict = {
                    "precision_micro": precision_micro,
                    "recall_micro": recall_micro,
                    "f1_micro": f1_micro,
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "f1_macro": f1_macro,
                }
                
                # Print metrics occasionally
                if i % 50 == 0:
                    print(f"Precision (micro): {precision_micro:.4f}, Recall (micro): {recall_micro:.4f}, F1 (micro): {f1_micro:.4f}")
                    print(f"Precision (macro): {precision_macro:.4f}, Recall (macro): {recall_macro:.4f}, F1 (macro): {f1_macro:.4f}")
                
                # Generate and log full classification report less frequently
                if i % 100 == 0:
                    # Get label names for the report
                    label_names = dataset.all_labels
                    report = get_classification_report(
                        output_probs, labels, threshold=0.5, labels=label_names
                    )
                    print("\nClassification Report:")
                    print(report)
                    writer.add_text('Classification Report', report, global_step)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', running_loss, global_step)
            writer.add_scalar('Jaccard/train', running_jaccard_index, global_step)
            
            # Log additional metrics
            for metric_name, metric_value in metrics_dict.items():
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, global_step)
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