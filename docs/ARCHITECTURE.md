# Architecture

## 1. Image representation

The `src/images` package models the data before learning begins.

It establishes the vocabulary used throughout the project:

- pixel and intensity
- grayscale and RGB
- image shape
- matrix and tensor representation
- normalization
- resizing
- inspection and visualization

## 2. Mathematical foundation

`src/math` contains the small linear-algebra layer used to explain how image values become manipulable numerical objects.

The progression is:

```
Vector
  ↓
Dot product
  ↓
Matrix
  ↓
Matrix multiplication
  ↓
Norm / distance
  ↓
Cosine similarity
  ↓
Linear transformation
```

The coordinate primitives make the geometric interpretation explicit without replacing the general vector/matrix implementation.

## 3. Vision primitives

`src/vision` implements convolution-oriented operations with explicit NumPy loops.

```
image
  ↓
padding
  ↓
sliding window
  ↓
element-wise multiplication
  ↓
sum
  ↓
output feature map
```

Both mathematical convolution and CNN-style cross-correlation are exposed so the distinction is visible rather than hidden.

## 4. Learning core

The `src/learning` package is the central bridge from features to learning.

```
Conv2D / Dense
      ↓
Activation
      ↓
Logits
      ↓
Cross-entropy
      ↓
dL / dlogits
      ↓
Backward pass
      ↓
Parameter gradients
      ↓
SGD
      ↓
Updated parameters
```

Each layer stores only the intermediate values required for its backward pass. There is no automatic differentiation engine.

## 5. Gradient verification

Gradient checks compare an analytical derivative with:

```
f(x + ε) - f(x - ε)
-------------------
        2ε
```

This is used to catch mistakes in the implementation of dense and convolutional gradients.

## 6. Data progression

The repository uses three levels of data:

1. tiny hand-built arrays for unit tests
2. deterministic synthetic images for learning experiments
3. external MNIST IDX files for a standard image classification task

This keeps tests small while still providing a path to a recognizable dataset.

## 7. Validation boundary

The core remains framework-free.

The optional PyTorch script exists only to answer a different question:

> Does the implementation agree numerically with a mature tensor framework on the same operation?

That separation keeps the educational implementation inspectable.
