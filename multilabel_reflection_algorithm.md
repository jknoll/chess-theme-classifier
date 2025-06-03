
# 📊 Step-by-Step Algorithm for Balancing Multi-Label Dataset via Targeted Reflections

This document describes an algorithm to improve label balance in a multi-label classification dataset by selectively duplicating underrepresented label sets through horizontal flipping (reflection). The objective is to bring the distribution of multi-label sets closer to uniformity.

---

## 🧠 Problem Setup

Let:

- `D = { (x₁, Y₁), ..., (xₙ, Yₙ) }` be your dataset, where each `Yᵢ ⊆ L` contains multiple labels from the label space `L`.
- `f(Y)` is the frequency of label set `Y` in the dataset.
- `R(Y)` is the set of augmentable samples with label set `Y`.

We want to select a subset of underrepresented `Y` and reflect samples from `R(Y)` such that the overall label set distribution becomes more uniform.

---

## 📊 Step-by-Step Algorithm

### Step 1: Count Multi-Label Frequencies

```python
from collections import Counter

# Assume labels is a list of label sets (as Python sets)
label_set_counts = Counter(frozenset(labels[i]) for i in range(len(labels)))
```

---

### Step 2: Compute Ideal Uniform Count

Let `M` be the number of unique label sets.

```python
ideal_count = total_samples / len(label_set_counts)
```

This represents the "target" count if all label sets were equally represented.

---

### Step 3: Calculate Gain from Reflection

For each label set `Y`, compute the L2 distance reduction gain from adding 1 more example:

\[
	ext{gain}(Y) = (f_Y - 	ext{ideal})^2 - (f_Y + 1 - 	ext{ideal})^2
\]

This measures how much duplicating `Y` improves balance.

```python
gains = {
    Y: (count - ideal_count)**2 - (count + 1 - ideal_count)**2
    for Y, count in label_set_counts.items()
}
sorted_Y = sorted(gains, key=gains.get, reverse=True)
```

---

### Step 4: Greedy Augmentation Loop

```python
budget = N  # Maximum number of reflections

reflections = []
for Y in sorted_Y:
    for sample in R(Y):
        if budget == 0:
            break
        reflections.append(reflect(sample))
        label_set_counts[Y] += 1
        budget -= 1
```

Optional constraints:
- Cap augmentation per label set (e.g., 3x original count).
- Avoid augmenting already frequent classes.

---

## ✅ Output

- A set of additional reflected samples.
- A new label set distribution closer to uniformity.

---

## 📚 References

- Chawla et al., *SMOTE: Synthetic Minority Over-sampling Technique*, 2002. [DOI:10.1613/jair.953](https://doi.org/10.1613/jair.953)
- Charte et al., *MLSMOTE: Multi-label imbalance recovery*, 2015. [DOI:10.1016/j.knosys.2015.03.013](https://doi.org/10.1016/j.knosys.2015.03.013)
