import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from model import Model, Lamb, get_lr_with_warmup
from dataset import ChessPuzzleDataset
from metrics import jaccard_similarity

class TestModelArchitecture:
    @pytest.fixture(scope="class")
    def model_setup(self):
        """Set up model with small dataset for testing"""
        # Use a very small subset for testing
        try:
            # Try to use the test dataset first
            if os.path.exists('lichess_db_puzzle_test.csv'):
                dataset = ChessPuzzleDataset('lichess_db_puzzle_test.csv')
            else:
                dataset = ChessPuzzleDataset('lichess_db_puzzle.csv')
                
            # Take only 10 samples for quick testing
            indices = list(range(10))
            subset = Subset(dataset, indices)
            
            # Create a dataloader with the subset
            dataloader = DataLoader(subset, batch_size=2, shuffle=True)
            
            # Get number of labels
            num_labels = len(dataset.all_labels)
            
            # Create model
            model = Model(num_labels=num_labels)
            
            # Create loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            
            # Use Lamb optimizer with settings from chess-hackathon (lr=0.001, weight_decay=0.01)
            lr = 0.001
            weight_decay = 0.01
            warmup_steps = 100  # Smaller for tests
            optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            return {
                "model": model,
                "dataloader": dataloader,
                "criterion": criterion,
                "optimizer": optimizer,
                "num_labels": num_labels,
                "lr_config": {
                    "base_lr": lr,
                    "warmup_steps": warmup_steps
                }
            }
        except Exception as e:
            pytest.fail(f"Failed to set up model and dataset: {str(e)}")

    def test_model_initialization(self, model_setup):
        """Test that the model initializes correctly"""
        model = model_setup["model"]
        assert isinstance(model, Model)
        assert hasattr(model, 'embedder')
        assert hasattr(model, 'convnet')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')
        assert model.num_labels == model_setup["num_labels"]

    def test_model_forward_pass(self, model_setup):
        """Test that a forward pass through the model works"""
        model = model_setup["model"]
        dataloader = model_setup["dataloader"]
        
        # Get one batch
        batch = next(iter(dataloader))
        inputs = batch['board'].unsqueeze(1)  # Add channel dimension
        
        # Run forward pass
        try:
            outputs = model(inputs)
            assert outputs.shape[0] == inputs.shape[0]  # Batch size same
            assert outputs.shape[1] == model_setup["num_labels"]  # Correct number of outputs
        except Exception as e:
            pytest.fail(f"Forward pass failed: {str(e)}")

    def test_model_debug_mode(self, model_setup):
        """Test that debug mode in forward pass works"""
        model = model_setup["model"]
        dataloader = model_setup["dataloader"]
        
        # Get one batch
        batch = next(iter(dataloader))
        inputs = batch['board'].unsqueeze(1)  # Add channel dimension
        
        # Run forward pass with debug=True
        try:
            outputs = model(inputs, debug=True)
            assert outputs.shape[0] == inputs.shape[0]
            assert outputs.shape[1] == model_setup["num_labels"]
        except Exception as e:
            pytest.fail(f"Forward pass with debug mode failed: {str(e)}")

    def test_model_backward_pass(self, model_setup):
        """Test that backward pass and optimization step work"""
        model = model_setup["model"]
        dataloader = model_setup["dataloader"]
        criterion = model_setup["criterion"]
        optimizer = model_setup["optimizer"]
        
        # Get one batch
        batch = next(iter(dataloader))
        inputs = batch['board'].unsqueeze(1)  # Add channel dimension
        labels = batch['themes']
        
        # Forward, backward, optimize
        try:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # If we get here without errors, the test passes
        except Exception as e:
            pytest.fail(f"Backward pass failed: {str(e)}")

    def test_model_training_loop(self, model_setup):
        """Test a mini training loop to verify end-to-end functionality"""
        model = model_setup["model"]
        dataloader = model_setup["dataloader"]
        criterion = model_setup["criterion"]
        optimizer = model_setup["optimizer"]
        lr_config = model_setup["lr_config"]
        
        # Run a mini training loop (2 epochs)
        try:
            losses = []
            jaccard_scores = []
            
            for epoch in range(2):
                epoch_loss = 0.0
                epoch_jaccard = 0.0
                num_batches = len(dataloader)
                step = 0
                
                for batch in dataloader:
                    inputs = batch['board'].unsqueeze(1).to('cpu')  # Add channel dimension
                    labels = batch['themes'].to('cpu')
                    
                    # Apply learning rate warmup
                    step_in_epoch = epoch * num_batches + step
                    lr = get_lr_with_warmup(
                        step=step_in_epoch,
                        total_steps=num_batches * 2,  # 2 epochs
                        warmup_steps=lr_config["warmup_steps"],
                        base_lr=lr_config["base_lr"]
                    )
                    
                    # Update learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Apply sigmoid to get probabilities for Jaccard calculation
                    output_probs = torch.sigmoid(outputs)
                    jaccard_loss = jaccard_similarity(output_probs, labels, threshold=0.5)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_jaccard += jaccard_loss.item()
                    step += 1
                
                avg_loss = epoch_loss / num_batches
                avg_jaccard = epoch_jaccard / num_batches
                losses.append(avg_loss)
                jaccard_scores.append(avg_jaccard)
            
            # Check that loss doesn't explode or become NaN
            for loss_val in losses:
                assert not np.isnan(loss_val), "Loss became NaN during training"
                assert not np.isinf(loss_val), "Loss became infinite during training"
            
            # Verify that Jaccard scores are valid (between 0 and 1)
            for jaccard_val in jaccard_scores:
                assert 0 <= jaccard_val <= 1, f"Jaccard score {jaccard_val} not between 0 and 1"
                
        except Exception as e:
            pytest.fail(f"Training loop failed: {str(e)}")

    def test_parameter_gradients(self, model_setup):
        """Test that gradients flow through all parts of the network"""
        model = model_setup["model"]
        dataloader = model_setup["dataloader"]
        criterion = model_setup["criterion"]
        
        # Get one batch
        batch = next(iter(dataloader))
        inputs = batch['board'].unsqueeze(1)  # Add channel dimension
        labels = batch['themes']
        
        # Zero all gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Check that gradients exist and are not all zero
        layers_with_grad = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and torch.any(param.grad != 0):
                    # Extract the top-level module name (before the first '.')
                    top_module = name.split('.')[0] if '.' in name else name
                    layers_with_grad.add(top_module)
        
        # These core components should definitely have gradients
        required_components = ['embedder', 'convLayers', 'accumulator', 'fc1', 'fc2', 'fc3']
        for component in required_components:
            assert component in layers_with_grad or any(comp.startswith(f"{component}.") for comp in layers_with_grad), \
                f"No gradients flowing through {component}"
                
    def test_lamb_optimizer(self, model_setup):
        """Test that the Lamb optimizer works correctly"""
        model = model_setup["model"]
        dataloader = model_setup["dataloader"]
        criterion = model_setup["criterion"]
        
        # Create new Lamb optimizer for this test
        optimizer = Lamb(
            model.parameters(), 
            lr=0.001, 
            weight_decay=0.01,
            eps=1e-6
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        inputs = batch['board'].unsqueeze(1)  # Add channel dimension
        labels = batch['themes']
        
        # Save initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.clone().detach()
        
        # Forward, backward, optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        params_updated = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Check if any parameter changed
                if not torch.allclose(param, initial_params[name]):
                    params_updated = True
                    break
        
        assert params_updated, "Lamb optimizer did not update any parameters"
        
    def test_lr_scheduler(self):
        """Test that the learning rate scheduler works correctly"""
        # Test warmup phase
        warmup_steps = 100
        base_lr = 0.001
        
        # Step 0 should return a very small lr
        lr_0 = get_lr_with_warmup(0, 1000, warmup_steps, base_lr)
        assert lr_0 == 0, "Learning rate should be 0 at step 0"
        
        # Step warmup_steps/2 should return about half the base lr
        lr_half = get_lr_with_warmup(warmup_steps // 2, 1000, warmup_steps, base_lr)
        assert 0.4 * base_lr <= lr_half <= 0.6 * base_lr, f"Learning rate at half warmup should be around half base_lr, got {lr_half}"
        
        # Step warmup_steps should return the base lr
        lr_full = get_lr_with_warmup(warmup_steps, 1000, warmup_steps, base_lr)
        assert lr_full == base_lr, f"Learning rate after warmup should be equal to base_lr, got {lr_full}"
        
        # Step > warmup_steps should also return the base lr
        lr_post = get_lr_with_warmup(warmup_steps + 50, 1000, warmup_steps, base_lr)
        assert lr_post == base_lr, f"Learning rate after warmup should be equal to base_lr, got {lr_post}"

if __name__ == "__main__":
    pytest.main()