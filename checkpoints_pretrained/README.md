The checkpoints in this directory are the pretrained models.

# checkpoint_resume.pth

This checkpoint was trained for 298,628 steps with the confifuration below. 

## Model configuration

```yaml
# Model configuration for chess-theme-classifier
# Based on the winning model architecture from chess-hackathon
nlayers: 5
embed_dim: 64
inner_dim: 320
attention_dim: 64
use_1x1conv: true
dropout: 0.5
# Number of labels from the full dataset
# Keep this at 1616 even when using the smaller test dataset
num_labels: 1616
```