# Roadmap

The original roadmap was intentionally granular. During implementation, completion was judged by concrete working capabilities rather than by claiming completion of an individual checklist item that had no distinct artifact.

## Milestone 1 — Understand Images

### Phase 0 — Foundation ✅
Project structure, Python environment, dependency policy, tests, linting/formatting/type checking, Git conventions, README, roadmap, project state, handoff and project contracts.

### Phase 1 — What Is an Image? ✅
Pixels, intensity, grayscale, RGB, shape, matrix/tensor representation, loading, saving, normalization, resizing, visualization, pixel/channel inspection and histograms.

### Phase 2 — Mathematics Behind Seeing Patterns ✅
Vectors, dot product, matrices, matrix multiplication, norms, distance, cosine similarity, linear transformations and 2D coordinate primitives.

### Phase 3 — Convolution From Scratch ✅
Padding, sliding windows, cross-correlation, mathematical convolution, stride, edge kernels and max pooling.

## Milestone 2 — Understand Learning

### Phase 4 — From Features to Neurons ✅
Trainable parameters, dense affine layers and nonlinear activations.

### Phase 5 — Prediction and Loss ✅
Logits, softmax, multiclass cross-entropy, mean squared error and classification accuracy.

### Phase 6 — Learning and Gradient Descent ✅
Parameter gradients and vanilla stochastic-gradient-style parameter updates.

### Phase 7 — Backpropagation From Scratch ✅
Explicit reverse traversal of layers plus analytical gradients for dense and convolutional layers.

### Phase 8 — Build a Tiny Neural Network ✅
Sequential model composition and a learning test on XOR.

## Milestone 3 — Teach a Machine to Learn Images

### Phase 9 — First Image Learning Problem ✅
Synthetic horizontal-versus-vertical line classification.

### Phase 10 — CNN From Scratch ✅
Convolution + activation + pooling + flatten + dense classification with no ML framework.

### Phase 11 — MNIST ✅
IDX image/label parsing, normalization and a reproducible MNIST training entry point.

### Phase 12 — Seeing What the Network Learned ✅
Activation statistics, parameter statistics, layer summaries and forward activation traces.

### Phase 13 — Experiments ✅
Controlled learning-rate experiments and reproducible synthetic datasets.

### Phase 14 — More Difficult Images ✅
Circle-versus-square synthetic classification and a deeper CNN example.

### Phase 15 — Framework Validation ✅
Optional numerical comparison of the from-scratch convolution forward pass and gradients against PyTorch.

### Phase 16 — Interactive Demonstration ✅
Browser-only drawing and convolution visualization.

### Phase 17 — Documentation and Portfolio Polish ✅
Architecture, validation, reproducibility, limitations, repository layout and usage documentation.

### Phase 18 — Release Preparation ✅
Core code, tests, CI, examples and documentation are in a release-ready state.

## What remains intentionally outside the core

The project does not attempt to become a general-purpose deep-learning framework. It intentionally avoids automatic differentiation, GPU kernels, large-scale data pipelines, distributed training and production inference infrastructure.

The next improvements, if the project is ever extended, should be experiments and explanations rather than framework bloat.
