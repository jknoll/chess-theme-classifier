When training locally, we are able to successfully write and read checkpoints. When training remotely on the ISC, the checkpoints are not written. ISC training is documented here:

https://docs.strongcompute.com/basic-concepts/launching-experiments

We are using the command `isc train chessVision.isc`

This command invokes the script train.py using the parameters in chessVision.isc.

Saving the checkpoint on the cluster requires using AtomicSavers as documented in the cycling_utils repo:

https://github.com/StrongResearch/cycling_utils/blob/c7ffeebc2b296296ea2ba3eada7dc733c86a5dda/README.md?plain=1#L18

There is a working implementation of a training script which successfully saves checkpoints on the ISC cluster here:

https://github.com/StrongResearch/chess-hackathon/blob/main/models/chessVision/train_chessVision.py

What are the critical differences between the current train.py implementation and this known-good working implementation?