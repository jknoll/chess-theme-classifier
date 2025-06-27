# Chess Theme Classifier - TODO

## High Priority

### Storage Optimization
- [ ] **Optimize tensor cache storage size** - [docs/optimize_tensor_cache_storage_size.md](docs/optimize_tensor_cache_storage_size.md)
  - Current: 31GB for 4.98M puzzles (float32 labels)
  - Target: 8.5GB with uint8 labels (75% reduction)
  - Impact: Faster loading, less memory usage, reduced storage costs
  - Effort: Low (single dtype change in dataset.py)

## Medium Priority

### Performance Improvements
- [ ] Implement bit-packed label storage for maximum compression (94% size reduction)
- [ ] Add sparse label storage option for puzzles with few active themes
- [ ] Benchmark training performance impact of uint8→float32 label conversion
- [ ] Profile memory usage during training with optimized storage

### Code Quality
- [ ] Add backward compatibility for existing float32 tensor caches
- [ ] Implement dual format support (uint8/float32) in dataset loader
- [ ] Add storage optimization parameter to dataset constructor
- [ ] Create migration scripts for existing tensor caches

## Low Priority

### Documentation
- [ ] Update training documentation to reflect storage optimization options
- [ ] Add performance benchmarks comparing storage formats
- [ ] Document migration path from float32 to uint8 labels

### Testing
- [ ] Add tests for uint8 label storage and conversion
- [ ] Verify correctness of bit-packed label operations
- [ ] Test loading performance across different storage formats

## Completed

### Dataset Generation
- [x] Milestone 1: Full dataset tensor cache (4.9M puzzles)
- [x] Milestone 2: Class-conditional augmented dataset
- [x] Implement single-threaded processing for container environments
- [x] Add progress monitoring and memory usage tracking
- [x] Create resume functionality for interrupted processing