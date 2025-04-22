import torch
import sys
import os
import importlib
import argparse

# First install torchinfo if not already installed
try:
    import torchinfo
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "torchinfo"])
    import torchinfo

def main():
    parser = argparse.ArgumentParser(description='Generate a summary of a chess model architecture')
    parser.add_argument('--model', type=str, default='model.Model', 
                        help='Model to summarize in format "module.ClassNameGoesHere". Default is "model.Model"')
    parser.add_argument('--num_labels', type=int, default=62,
                        help='Number of output labels for the model. Default is 62')
    args = parser.parse_args()
    
    # Parse the module and class name
    try:
        module_name, class_name = args.model.rsplit('.', 1)
    except ValueError:
        # If there's no dot, assume it's a class in the current directory
        module_name = args.model
        class_name = args.model
    
    # Import the module and get the class
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        print(f"Model initialized from {module_name}.{class_name}")
    except (ImportError, AttributeError) as e:
        print(f"Error loading model {args.model}: {e}")
        if '.' not in args.model:
            print(f"Did you mean 'model.{args.model}' or 'original_model.{args.model}'?")
        sys.exit(1)
    
    # Create the model
    model = model_class(num_labels=args.num_labels)
    
    # Generate dummy input based on model type
    if module_name == 'original_model':
        # OriginalModel expects a (N, C, H, W) tensor with float values
        # For CNNs, we need a 4D tensor with batch_size, channels, height, width
        # It expects 1 channel for grayscale images
        dummy_input = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    else:
        # Default model expects integers representing chess pieces (0-12)
        dummy_input = torch.randint(0, 13, (1, 8, 8), dtype=torch.long)
    
    try:
        # Generate and print the summary
        summary = torchinfo.summary(
            model, 
            input_data=dummy_input,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=4,
            verbose=2
        )
        
        print(summary)
    except RuntimeError as e:
        print(f"Error generating model summary: {e}")
        print("\nTrying alternative input format...")
        
        # Try the other input format as a fallback
        if module_name == 'original_model':
            dummy_input = torch.randint(0, 13, (1, 8, 8), dtype=torch.long)
        else:
            dummy_input = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        
        try:
            summary = torchinfo.summary(
                model, 
                input_data=dummy_input,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                depth=4,
                verbose=2
            )
            print(summary)
        except Exception as e2:
            print(f"Failed with alternative input format as well: {e2}")
            print("Please check the model's forward method to understand required input format.")
            sys.exit(1)

if __name__ == "__main__":
    main()