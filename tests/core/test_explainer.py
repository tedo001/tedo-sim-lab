"""CNN Explainer engine: exact against PyTorch, and every value it takes apart adds up."""

from __future__ import annotations

import numpy as np
import pytest

from labs.computer_vision.explainer import (
    PRESETS,
    ArchitectureError,
    ConvGeometry,
    ExplainerNet,
    LayerSpec,
    conv_intermediates,
    conv_output_size,
    conv_step,
    linear_contributions,
    map_limits,
    pool_step,
    prepare_image,
    run,
    softmax_terms,
    tiny_vgg,
)
from labs.computer_vision.explainer.network import Architecture
from labs.computer_vision.explainer.samples import digit_samples, pattern_samples, resize_bilinear

NAMES = tuple(str(i) for i in range(10))


@pytest.fixture(scope="module")
def trace():
    net = ExplainerNet.untrained(tiny_vgg(in_channels=3, size=32, class_names=NAMES), seed=3)
    image = np.random.default_rng(0).uniform(size=(3, 32, 32))
    return run(net, image)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_numpy_forward_matches_pytorch(preset) -> None:
    torch = pytest.importorskip("torch")
    from labs.computer_vision.models import to_torch
    architecture = preset.architecture()
    net = ExplainerNet.untrained(architecture, seed=1)
    model = to_torch(architecture)
    model.load_state_dict({name: torch.tensor(value, dtype=torch.float32)
                           for name, value in net.params.items()})
    image = preset.samples()[0].image
    with torch.no_grad():
        expected = model(torch.tensor(image[None], dtype=torch.float32))[0].numpy()
    np.testing.assert_allclose(run(net, image).logits, expected, atol=1e-5)


def test_tiny_vgg_shapes() -> None:
    architecture = tiny_vgg(in_channels=1, size=28, class_names=NAMES)
    assert architecture.shapes[0] == (10, 26, 26) and architecture.shapes[4] == (10, 12, 12)
    assert architecture.shapes[9] == (10, 4, 4) and architecture.shapes[10] == (160,)
    assert architecture.param_shapes["conv_1_1.weight"] == (10, 1, 3, 3)
    assert architecture.param_shapes["output.weight"] == (10, 160)


def test_architecture_errors_are_explained() -> None:
    with pytest.raises(ArchitectureError, match="does not fit"):
        tiny_vgg(in_channels=1, size=12, class_names=NAMES)
    with pytest.raises(ArchitectureError, match="flatten"):
        Architecture("x", (1, 8, 8), (LayerSpec("fc", "linear", channels=2),), ("a", "b"))
    with pytest.raises(ArchitectureError, match="class names"):
        tiny_vgg(in_channels=1, size=28, class_names=NAMES).__class__(
            "x", (1, 8, 8), (LayerSpec("f", "flatten"), LayerSpec("o", "linear", channels=3)), ("a",))
    assert conv_output_size(5, 3, stride=2, padding=1) == 3


def test_weights_are_checked_and_round_trip(tmp_path) -> None:
    architecture = tiny_vgg(in_channels=1, size=28, class_names=NAMES)
    net = ExplainerNet.untrained(architecture, seed=7)
    assert net.source == "random, seed 7" and not net.trained
    again = ExplainerNet.untrained(architecture, seed=7)
    assert all(np.array_equal(net.params[k], again.params[k]) for k in net.params)
    bound = 1 / np.sqrt(9)  # conv_1_1: one input channel, 3×3
    assert np.abs(net.weight("conv_1_1")).max() <= bound
    net.save_npz(tmp_path / "w.npz")
    loaded = ExplainerNet.load_npz(architecture, tmp_path / "w.npz")
    assert loaded.trained and np.allclose(loaded.weight("output"), net.weight("output"), atol=1e-6)
    broken = dict(net.params)
    broken["conv_1_1.weight"] = np.zeros((10, 3, 3, 3))
    with pytest.raises(ArchitectureError, match="conv_1_1.weight has shape"):
        ExplainerNet(architecture, broken)
    del broken["output.bias"]
    with pytest.raises(ArchitectureError, match="missing"):
        ExplainerNet(architecture, broken)


def test_conv_step_adds_up(trace) -> None:
    index = trace.net.architecture.index_of("conv_1_2")
    step = conv_step(trace, index, channel=4, y=6, x=11)
    assert step.windows.shape == (10, 3, 3) and step.origin == (6, 11)
    assert step.value == pytest.approx(trace.outputs[index][4, 6, 11])
    assert step.partials.sum() + step.bias == pytest.approx(step.value)
    intermediates = conv_intermediates(trace, index, 4)
    np.testing.assert_allclose(intermediates[:, 6, 11], step.partials)
    np.testing.assert_allclose(intermediates.sum(0) + step.bias, trace.outputs[index][4])


def test_conv_step_with_padding_and_stride() -> None:
    layers = (LayerSpec("conv", "conv", channels=2, kernel=3, stride=2, padding=1),
              LayerSpec("flatten", "flatten"), LayerSpec("output", "linear", channels=2))
    net = ExplainerNet.untrained(Architecture("p", (1, 5, 5), layers, ("a", "b")), seed=2)
    trace = run(net, np.arange(25.0).reshape(1, 5, 5) / 25)
    assert trace.outputs[0].shape == (2, 3, 3)
    corner = conv_step(trace, 0, 1, 0, 0)
    assert corner.origin == (-1, -1) and corner.windows[0, 0].tolist() == [0, 0, 0]
    assert corner.value == pytest.approx(trace.outputs[0][1, 0, 0])
    assert conv_step(trace, 0, 0, 2, 2).value == pytest.approx(trace.outputs[0][0, 2, 2])


def test_pool_relu_softmax_and_linear(trace) -> None:
    pool = trace.net.architecture.index_of("max_pool_1")
    step = pool_step(trace, pool, 2, 3, 5)
    assert step.origin == (6, 10) and step.value == pytest.approx(trace.outputs[pool][2, 3, 5])
    assert step.window[step.argmax] == step.window.max()
    relu = trace.net.architecture.index_of("relu_1_1")
    assert (trace.outputs[relu] >= 0).all()
    np.testing.assert_array_equal(trace.outputs[relu], np.maximum(trace.outputs[relu - 1], 0))
    terms = softmax_terms(trace.logits)
    assert sum(terms.probability(i) for i in range(10)) == pytest.approx(1)
    assert terms.probability(trace.prediction) == pytest.approx(trace.probabilities.max())
    big = softmax_terms(np.array([1000.0, 999.0]))  # would overflow exp() unshifted
    assert np.isfinite(big.total) and big.probability(0) == pytest.approx(1 / (1 + np.exp(-1)))
    output = trace.net.architecture.index_of("output")
    contributions, bias = linear_contributions(trace, output, 7)
    assert contributions.shape == (10, 5, 5)
    assert contributions.sum() + bias == pytest.approx(trace.logits[7])


def test_colour_ranges_follow_cnn_explainer(trace) -> None:
    layers = trace.net.architecture.layers
    local = map_limits(trace, "layer")
    assert set(local) == {i for i, layer in enumerate(layers) if layer.is_map}
    conv, relu, pool = (trace.net.architecture.index_of(n) for n in ("conv_1_2", "relu_1_2", "max_pool_1"))
    assert local[conv] == local[relu] == local[pool] == pytest.approx(np.abs(trace.outputs[conv]).max())
    block = map_limits(trace, "block")
    assert block[0] == block[pool] == max(local[i] for i in range(pool + 1))
    network = map_limits(trace, "network")
    assert len(set(network.values())) == 1 and network[0] == max(local.values())


def test_playground_geometry() -> None:
    geometry = ConvGeometry(5, 3, padding=1, stride=2)
    assert geometry.padded == 7 and geometry.output == 3 and geometry.leftover == 0
    assert geometry.window(1, 2) == (2, 4) and geometry.formula().endswith("= 3")
    uneven = ConvGeometry(6, 3, stride=2)
    assert uneven.output == 2 and uneven.leftover == 1
    assert uneven.coverage()[:, -1].sum() == 0 and uneven.coverage()[:5, :5].all()
    assert geometry.is_padding(0, 3) and not geometry.is_padding(1, 1)
    fitted = ConvGeometry.fitted(input=4, kernel=9, padding=5, stride=9)
    assert fitted.padding < fitted.kernel <= fitted.padded and fitted.stride <= fitted.max_stride


def test_inputs() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    grey = prepare_image(rgb, 1)
    assert grey.shape == (1, 4, 4) and grey[0, 0, 0] == pytest.approx(0.299)
    assert prepare_image(rgb, 3)[0].min() == 1
    assert prepare_image(rgb, 1, invert=True)[0, 0, 0] == pytest.approx(0.701)
    assert resize_bilinear(np.eye(2), 4, 4).shape == (4, 4)
    assert len(pattern_samples(28, 1)) == 6
    assert all(s.image.shape == (3, 32, 32) for s in pattern_samples(32, 3))
    digits = digit_samples(28)
    if not digits:
        pytest.skip("scikit-learn (which ships the digits) is not installed")
    assert [d.label for d in digits] == list(range(10)) and digits[3].image.shape == (1, 28, 28)
    assert digits[3].image[0, :4].max() == 0 and digits[3].image.max() <= 1  # framed like MNIST
