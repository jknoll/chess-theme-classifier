# Adaptive Thresholding for Model Classification Evaluation 

We have implemented code to change `evaluate_model_classification` so that it uses the same adaptive thresholding mechanism described in adaptive_thresholding.md.

1) [x] Weights-only deserialization fails; fixed temporarily by setting weights-only=False.
2) [x] Error attempting to deserialize on a CUDA device when container started with 0 GPUs. Restarting container.
3) [x] Show tqdm progress conditionally to increase token efficiency.
4) [x] Should args.optimized be the default? Optimized mode is now the default, and we use --no_optimize flag to disable it.
5) [x] We are using the test.csv to extract labels for some reason. Added --use_cache flag to use cached tensors directly.
6) [x] Fixed issue with `flat_outputs` variable in the `calculate_adaptive_threshold` function when using the `--use_cache` option.