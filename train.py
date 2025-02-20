from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

from dataset import ChessPuzzleDataset
import torch
from torch.utils.data import DataLoader, random_split
from model import Model

def main():
    # Set up the device for GPU use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)


    model = Model()
    model = model.to(device)

    dataset = ChessPuzzleDataset('lichess_db_puzzle.csv')

    # Get a single item
    sample = dataset[0]
    print(f"Example dataset item: {sample}")
 #   print(f"Theme names: {dataset.get_theme_names()}")
 #   print(f"Theme count: {len(dataset.get_theme_names())}")

    input = sample['board'].unsqueeze(0).unsqueeze(0) # More elegant way to do this?

    model = Model()
    print(model)

    print(input)
    out = model(input)
    print(f"Out: {out}")

    target = sample['themes'].unsqueeze(0)
    print(f"Target: {target}")
    # Define the loss function and optimizer
    criterion = torch.nn.BCELoss() # Or should we use BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss = criterion(out, target)
    print(f"Loss: {loss}")

    ############################################################

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

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['board']  # Shape: [batch_size, height, width]
            inputs = inputs.unsqueeze(1)  # Add channel dimension: [batch_size, channels=1, height, width]
            labels = data['themes']  # Remove the unsqueeze since batch dimension is already there

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(f"epoch: {epoch} running loss: {running_loss/(i+1)} examples: {i+1}")
            # Save checkpoint
            if i % 100000 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, "checkpoint.pth")

if __name__ == "__main__":
    main()