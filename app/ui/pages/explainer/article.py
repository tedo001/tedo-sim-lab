"""The explainer's reading text, adapted from the CNN Explainer article (Wang et al.,
2020; MIT licence) for the lab's networks."""

from __future__ import annotations

__all__ = ["ARTICLE", "CREDIT"]

CREDIT = (
    "Adapted from CNN Explainer by Zijie J. Wang, Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, "
    "Fred Hohman, Minsuk Kahng and Duen Horng (Polo Club) Chau, Polo Club of Data Science, Georgia Tech "
    "(IEEE VIS 2020). MIT licence. CNN Explainer's Tiny ImageNet images and pretrained weights are not "
    "used: they derive from ImageNet."
)

ARTICLE = """\
### What is a convolutional network?

A classifier assigns a label to an input; an image classifier says what an image shows. A
convolutional neural network (CNN) is a classifier built from layers of neurons, each with
weights and a bias that are learned during training. Its special ingredient is the
**convolutional layer**, which looks at small neighbourhoods of pixels and is good at finding
patterns in images.

The network here is **TinyVGG**, the network CNN Explainer uses: two blocks of two 3×3
convolutions with ReLU, each block ending in a 2×2 max-pool, then one linear layer that scores
every class. It is small enough to see completely, and it uses the same operations as large
modern networks.

### Input

The input layer is the image: one channel for a grey image, three (red, green, blue) for a
colour one, each pixel a number from 0 to 1.

### Convolution

A convolutional neuron has one small kernel (here 3×3) for every channel of the previous layer.
Each kernel slides over its channel; at every position the window and the kernel are multiplied
element by element and summed. That gives one intermediate map per input channel. Adding the
intermediate maps together, plus the neuron's bias, gives its **activation map**. Because every
neuron in TinyVGG connects to every channel before it, the first layer of a colour network has
3 × 10 = 30 kernels.

### Hyperparameters

- **Padding** surrounds the input with zeros so the kernel can centre on edge pixels; it keeps
  information at the borders and can keep the map's size.
- **Kernel size** is the size of the sliding window. Small kernels capture local detail and shrink
  the map slowly, which allows deeper networks.
- **Stride** is how far the kernel moves each step. A larger stride gives a smaller output and
  extracts less.

The output size is ⌊(n + 2p − k) / s⌋ + 1 for an n×n input, padding p, kernel k and stride s.

### ReLU

ReLU, max(0, x), keeps positive values and turns negative ones into zero. This non-linearity is
what lets a deep network learn more than a single layer could; without it, stacked
convolutions collapse into one. ReLU is also quick to train, which is why it replaced
functions such as the sigmoid.

### Max-pooling

Pooling shrinks the maps. Max-pooling slides a window (2×2, stride 2 here) and keeps the largest
value in each, discarding three values in four. That cuts the computation in the layers that
follow and makes the network less sensitive to small shifts.

### Flatten, scores and softmax

Flatten lays the last maps out as one long vector. A linear layer gives every class a
**score** (a logit): a weighted sum of the whole vector plus a bias. **Softmax** turns the
scores into probabilities: exp(z) for a class divided by the sum of exp(z) over all classes. The
results are positive, add up to 1, and favour the largest score more than plain rescaling
would, a "soft" version of picking the maximum that can still be trained.

### Reading this page

- Click a sample, open an image, or draw a digit to change the input.
- Hover a map in the network to see which maps feed it; click it to see the arithmetic.
- In a detail view, hover a map to move the kernel, or press Play to animate it.
- *Colour scale* sets whether each layer, each block or the whole network shares one range.
  Red is negative, blue positive, grey zero.
"""
