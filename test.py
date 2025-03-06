import torch
from dataset import ChessPuzzleDataset
from model import Model
from metrics import jaccard_similarity

# Create model instance
model = Model()

# Load the checkpoint and extract just the model state dict
checkpoint = torch.load("checkpoint_resume.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

# Set model to evaluation mode
model.eval()

dataset = ChessPuzzleDataset("lichess_db_puzzle.csv")

# Get theme names
theme_names = dataset.get_theme_names()

# Process first 10 samples
for i in range(10):
    sample = dataset[i]
    input = sample['board'].unsqueeze(0).unsqueeze(0)
    target = sample['themes']

    with torch.no_grad():
        out = model(input)
        out = torch.sigmoid(out)  # Convert logits to probabilities

    # Calculate Jaccard index on the raw probabilities vs thresholded predictions
    raw_jaccard = jaccard_similarity(out.squeeze(), target, threshold=0.5)

    # Get predicted themes (where probability > 0.5)
    predicted_probs, predicted_indices = torch.where(out > 0.5, out, torch.zeros_like(out)).squeeze().sort(descending=True)
    predicted_themes = [(theme_names[idx], f"{predicted_probs[i]:.3f}") for i, idx in enumerate(predicted_indices) if predicted_probs[i] > 0.5]
    
    # Get the theme names only from predicted themes (without probabilities)
    predicted_theme_names = [theme for theme, _ in predicted_themes]

    # Get actual themes
    actual_themes = [theme_names[i] for i, is_theme in enumerate(target) if is_theme == 1]

    # Calculate Jaccard index using string lists (should match raw_jaccard)
    name_jaccard = jaccard_similarity(predicted_theme_names, actual_themes)

    print(f"\n=== Sample {i+1} ===")
    print(f"\nJaccard Index (using thresholded probabilities): {raw_jaccard:.3f}")
    print(f"Jaccard Index (using theme name lists): {name_jaccard:.3f}")

    print("\nPredicted themes (probability):")
    for theme, prob in predicted_themes:
        print(f"{theme}: {prob}")

    print("\nActual themes:")
    print(", ".join(actual_themes))

    print("\nPosition FEN:")
    print(sample['fen'])
    print("="*50)