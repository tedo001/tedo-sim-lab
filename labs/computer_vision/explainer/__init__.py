"""CNN Explainer: what a convolutional network computes, layer by layer.

A port of CNN Explainer by Zijie J. Wang, Robert Turko, Omar Shaikh, Haekyu Park,
Nilaksh Das, Fred Hohman, Minsuk Kahng and Duen Horng (Polo Club) Chau
(Polo Club of Data Science, Georgia Tech; https://github.com/poloclub/cnn-explainer),
MIT licence, see ``LICENSE-cnn-explainer.txt``. The computation lives here in numpy
(no Qt, no PyTorch), so the interface can run it instantly and show every number.
"""

from .forward import (
    ConvStep,
    PoolStep,
    SoftmaxTerms,
    Trace,
    conv_intermediates,
    conv_step,
    linear_contributions,
    pool_step,
    run,
    softmax_terms,
)
from .geometry import ConvGeometry
from .network import Architecture, ArchitectureError, ExplainerNet, LayerSpec, conv_output_size, tiny_vgg
from .presets import PRESETS, Preset, preset
from .samples import Sample, prepare_image
from .scales import SCALES, Scale, map_limits

__all__ = ["PRESETS", "SCALES", "Architecture", "ArchitectureError", "ConvGeometry", "ConvStep",
           "ExplainerNet", "LayerSpec", "PoolStep", "Preset", "Sample", "Scale", "SoftmaxTerms", "Trace",
           "conv_intermediates", "conv_output_size", "conv_step", "linear_contributions", "map_limits",
           "pool_step", "prepare_image", "preset", "run", "softmax_terms", "tiny_vgg"]
