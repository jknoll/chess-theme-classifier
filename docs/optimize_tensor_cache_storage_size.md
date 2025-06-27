# Optimize Tensor Cache Storage Size

## Current Issue

The tensor cache files are significantly larger than necessary due to inefficient label storage. Analysis of the class-conditional augmented dataset reveals:

- **Current file size**: 31GB for 4.98M puzzles
- **Label storage**: 1,616 themes × 4 bytes (float32) = 6,464 bytes per puzzle
- **Label content**: Pure binary values (0.0 or 1.0 only)
- **Storage waste**: 75% of file size is due to inefficient label encoding

## Storage Breakdown

| Component | Current Size | Optimized Size | Savings |
|-----------|--------------|----------------|---------|
| Board tensors (8×8 int8) | 64 bytes/puzzle | 64 bytes/puzzle | 0% |
| Labels (1,616 float32) | 6,464 bytes/puzzle | - | - |
| Labels (1,616 uint8) | - | 1,616 bytes/puzzle | 75% |
| Labels (bit-packed) | - | 202 bytes/puzzle | 97% |
| **Total per puzzle** | **6,528 bytes** | **1,680 bytes (uint8)** | **75%** |
| **Total dataset** | **31GB** | **8.5GB (uint8)** | **75%** |

## Optimization Options

### Option 1: Convert to uint8 (Recommended)
**Implementation**: Change label dtype from float32 to uint8
- **File size reduction**: 31GB → 8.5GB (75% savings)
- **Memory reduction**: 75% less RAM during training
- **Compatibility**: Requires conversion to float32 during training
- **Complexity**: Low - single dtype change

```python
# Current (wasteful):
label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)

# Optimized:
label_vector = torch.zeros(len(self.all_labels), dtype=torch.uint8)
```

**Training adaptation needed**:
```python
# In DataLoader/training loop:
labels = labels.float()  # Convert uint8 → float32 for loss functions
```

### Option 2: Bit Packing (Maximum Optimization)
**Implementation**: Pack 8 binary labels per byte
- **File size reduction**: 31GB → 2.4GB (94% savings)  
- **Storage**: 1,616 bits = 202 bytes per puzzle
- **Complexity**: High - requires bit packing/unpacking logic
- **Performance**: Slight CPU overhead for bit operations

```python
# Bit packing example:
def pack_binary_labels(labels):
    """Pack binary labels into bits, 8 labels per byte"""
    # labels: tensor of 0s and 1s
    packed = torch.zeros((len(labels) + 7) // 8, dtype=torch.uint8)
    for i, label in enumerate(labels):
        if label:
            packed[i // 8] |= (1 << (i % 8))
    return packed

def unpack_binary_labels(packed, num_labels):
    """Unpack bit-packed labels back to binary tensor"""
    labels = torch.zeros(num_labels, dtype=torch.uint8)
    for i in range(num_labels):
        labels[i] = (packed[i // 8] >> (i % 8)) & 1
    return labels
```

### Option 3: Sparse Label Storage
**Implementation**: Store only active label indices
- **File size reduction**: Variable (depends on label sparsity)
- **Average labels per puzzle**: ~3-5 active out of 1,616 total
- **Storage**: ~20 bytes per puzzle for label indices
- **Complexity**: Medium - requires sparse format handling

```python
# Sparse format example:
{
    'board_tensors': tensor,  # [N, 8, 8] int8
    'active_label_indices': [  # List of lists
        [45, 123, 892],      # Puzzle 0 has themes 45, 123, 892
        [12, 67],            # Puzzle 1 has themes 12, 67
        # ...
    ]
}
```

## Code Locations Requiring Changes

The following locations in `dataset.py` create float32 label vectors:

1. **Line 1226**: `label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)`
2. **Line 1263**: `reflection_label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)`  
3. **Line 1526**: `label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)`

## Benefits of Optimization

### Storage Benefits
- **Disk space**: 75-94% reduction in file size
- **Backup/transfer**: Faster data movement
- **Storage costs**: Significant reduction in cloud storage fees

### Performance Benefits  
- **Loading speed**: 75% faster dataset loading from disk
- **Memory usage**: 75% less RAM for label data
- **Distributed training**: 75% less network transfer
- **Cache efficiency**: More data fits in memory caches

### Practical Impact
- **31GB → 8.5GB**: Fits on smaller storage devices
- **Loading time**: 30 seconds → 8 seconds for full dataset
- **Memory footprint**: Critical for systems with limited RAM
- **Multi-GPU training**: Reduced memory pressure per GPU

## Implementation Recommendations

1. **Start with Option 1 (uint8)**: Best effort-to-benefit ratio
2. **Measure performance impact**: Benchmark training speed with uint8→float32 conversion
3. **Consider Option 2 (bit packing)**: If storage is critical and CPU overhead acceptable
4. **Backward compatibility**: Maintain ability to load existing float32 caches

## Testing Plan

1. **Create optimized cache**: Generate uint8 version of existing dataset
2. **Verify correctness**: Ensure labels identical after conversion
3. **Benchmark training**: Compare training speed with uint8 vs float32 labels
4. **Measure loading**: Compare dataset loading times
5. **Memory profiling**: Confirm memory usage reduction

## Related Files

- `dataset.py`: Label tensor creation and caching logic
- `train.py`: Training loop that consumes labels  
- `create_full_dataset_cache.py`: Basic tensor cache generation
- `create_full_class_conditional_dataset.py`: Augmented dataset generation

## Migration Strategy

1. **Add new parameter**: `optimize_storage=True` to dataset constructor
2. **Dual format support**: Support both uint8 and float32 label loading
3. **Gradual migration**: Generate new optimized caches while maintaining old format compatibility
4. **Training adaptation**: Automatic conversion in `__getitem__` method