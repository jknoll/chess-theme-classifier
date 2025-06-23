"""
Utilities for handling model checkpoint operations in different training environments.

This module provides a unified interface for saving and loading model checkpoints
across different environments:
- Local dual GPU workstation (using data parallel mode)
- Remote container with multiple GPUs
- Training cluster

The module detects the appropriate format based on the checkpoint structure and
handles the differences transparently.
"""

import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_distributed_model(model):
    """
    Check if the model is wrapped in DistributedDataParallel or DataParallel.
    
    Args:
        model: The PyTorch model to check
        
    Returns:
        bool: True if the model is distributed, False otherwise
    """
    return isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel))

def get_model_state_dict(model):
    """
    Get the state dictionary from a model, handling distributed models appropriately.
    
    Args:
        model: The PyTorch model
        
    Returns:
        dict: The model's state dictionary
    """
    if is_distributed_model(model):
        return model.module.state_dict()
    else:
        return model.state_dict()

def save_checkpoint(model, optimizer, epoch, loss, global_step, output_path, 
                   additional_data=None, is_master=True, filename="checkpoint_resume.pth"):
    """
    Save a model checkpoint with consistent formatting across environments.
    
    Args:
        model: The PyTorch model to save
        optimizer: The optimizer
        epoch: Current epoch number
        loss: Current loss value
        global_step: Global training step
        output_path: Directory to save the checkpoint
        additional_data: Optional dictionary of additional data to include
        is_master: Whether this process is the master process (only save if True)
        filename: Name of the checkpoint file
        
    Returns:
        str: Path to the saved checkpoint file or None if not saved
    """
    if not is_master:
        logger.debug("Not saving checkpoint - not the master process")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get the model state dict, handling distributed models
    model_state = get_model_state_dict(model)
    
    # Prepare checkpoint data in a consistent format
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'global_step': global_step,
    }
    
    # Add additional data if provided
    if additional_data:
        checkpoint_data.update(additional_data)
    
    # Save the checkpoint
    checkpoint_path = os.path.join(output_path, filename)
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, device=None, strict=False):
    """
    Load a model checkpoint, automatically detecting the format.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        device: Optional device to map the checkpoint to
        strict: Whether to strictly enforce that the keys in state_dict match the keys
               returned by this module's state_dict() function. Default is False to allow
               loading checkpoints between models with different architectures.
        
    Returns:
        dict: The loaded checkpoint data
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If there's an error loading the checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # If device is provided, map checkpoint to that device
    if device:
        map_location = device
    else:
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading checkpoint from {checkpoint_path} to {map_location}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")
    
    # Detect checkpoint format based on keys
    # Different training environments may use different keys
    if 'model_state_dict' in checkpoint:
        # Standard format
        model_state_key = 'model_state_dict'
        optimizer_state_key = 'optimizer_state_dict'
        logger.info("Detected standard checkpoint format")
    elif 'model' in checkpoint:
        # ISC/Cluster format
        model_state_key = 'model'
        optimizer_state_key = 'optimizer'
        logger.info("Detected ISC/Cluster checkpoint format")
    else:
        raise RuntimeError("Unknown checkpoint format. Expected 'model_state_dict' or 'model' key.")
    
    # Load model state, handling distributed models
    try:
        if is_distributed_model(model):
            result = model.module.load_state_dict(checkpoint[model_state_key], strict=strict)
        else:
            result = model.load_state_dict(checkpoint[model_state_key], strict=strict)
            
        # Log information about missing/unexpected keys when strict=False
        if not strict and (result.missing_keys or result.unexpected_keys):
            logger.info(f"Non-strict loading detected. Model and checkpoint have different structures:")
            
            if result.missing_keys:
                logger.info(f"Missing {len(result.missing_keys)} keys in model that are in checkpoint")
                if len(result.missing_keys) < 10:
                    logger.info(f"Missing keys: {result.missing_keys}")
                else:
                    logger.info(f"First 10 missing keys: {result.missing_keys[:10]}")
                    
            if result.unexpected_keys:
                logger.info(f"Unexpected {len(result.unexpected_keys)} keys in model that are not in checkpoint")
                if len(result.unexpected_keys) < 10:
                    logger.info(f"Unexpected keys: {result.unexpected_keys}")
                else:
                    logger.info(f"First 10 unexpected keys: {result.unexpected_keys[:10]}")
    except Exception as e:
        logger.error(f"Error loading model state: {str(e)}")
        if strict:
            logger.info("Consider using strict=False to allow loading checkpoints with different architectures")
        raise
    
    # Load optimizer state if provided
    if optimizer is not None and optimizer_state_key in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint[optimizer_state_key])
            logger.info("Loaded optimizer state")
        except Exception as e:
            logger.warning(f"Error loading optimizer state: {str(e)}")
    
    # Return information from the checkpoint
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'loss': checkpoint.get('loss', None),
    }
    
    logger.info(f"Checkpoint loaded successfully. Epoch: {checkpoint_info['epoch']}, "
                f"Global step: {checkpoint_info['global_step']}")
    
    return checkpoint_info

def get_checkpoint_info(checkpoint_path, device=None):
    """
    Get information about a checkpoint without loading it into a model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Optional device to map the checkpoint to
        
    Returns:
        dict: Checkpoint information
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # If device is provided, map checkpoint to that device
    if device:
        map_location = device
    else:
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")
    
    # Extract basic information
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'loss': checkpoint.get('loss', None),
        'format': 'unknown',
    }
    
    # Detect format
    if 'model_state_dict' in checkpoint:
        info['format'] = 'standard'
    elif 'model' in checkpoint:
        info['format'] = 'isc_cluster'
    
    # Add additional info if available
    if 'learning_rate' in checkpoint:
        info['learning_rate'] = checkpoint['learning_rate']
    if 'jaccard_loss' in checkpoint:
        info['jaccard_loss'] = checkpoint['jaccard_loss']
    
    return info