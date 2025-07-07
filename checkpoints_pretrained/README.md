The checkpoints in this directory are the pretrained models.

# checkpoint_resume.pth

This checkpoint was trained for 298,628 steps with the configuration below. This was supposedly some large number of epochs, 40(?). But I have now verified that cached tensor file derived from the full size .csv did not contain all board positions (plus the class-balancing additions), but rather a small subset plus the class-balancing additions, approximately 65K total positions.

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

# AtomicDirectory_checkpoint_2428

This checkpoint was trained on the ISC with the full class-conditional dataset (about 4.9M positions, with an 0.85 train split and a 0.15 test split) for about 150K steps.

## Model configuration
```yaml
# Model configuration for chess-theme-classifier
# Based on the winning model architecture from chess-hackathon
nlayers: 10
embed_dim: 64
inner_dim: 320
attention_dim: 64
use_1x1conv: true
dropout: 0.5
# Number of labels from the full dataset
# Keep this at 1616 even when using the smaller test dataset
num_labels: 1616
```

# AtomicDirectory_checkpoint_5756

This checkpoint was continuation-trained on the ISC after AtomicDirectory_checkpoint_2428 with the full class-conditional dataset (about 4.9M positions, with an 0.85 train split and a 0.15 test split) for more steps.

## Model configuration
```yaml
# Model configuration for chess-theme-classifier
# Based on the winning model architecture from chess-hackathon
nlayers: 10
embed_dim: 64
inner_dim: 320
attention_dim: 64
use_1x1conv: true
dropout: 0.5
# Number of labels from the full dataset
# Keep this at 1616 even when using the smaller test dataset
num_labels: 1616
```

# AtomicDirectory_checkpoint_319

This checkpoint was a continuation of the training run of AtomicDirectory_checkpoint_5756 after the training job failed.

## Model configuration
```yaml
# Model configuration for chess-theme-classifier
# Based on the winning model architecture from chess-hackathon
nlayers: 10
embed_dim: 64
inner_dim: 320
attention_dim: 64
use_1x1conv: true
dropout: 0.5
# Number of labels from the full dataset
# Keep this at 1616 even when using the smaller test dataset
num_labels: 1616
```