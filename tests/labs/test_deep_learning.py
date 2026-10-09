"""The layer-stack planner (shapes, parameters, errors, generated code) and the PyTorch module
built from the same layers."""

from __future__ import annotations

import pytest

from labs.deep_learning.layers import LayerError, plan, pytorch_code, starter_stack


def test_starter_stack_on_mnist_and_cifar() -> None:
    steps = plan(starter_stack(), in_channels=1, input_size=(28, 28), num_classes=10)
    assert steps[3].shape == (16, 14, 14) and steps[7].shape == (32, 7, 7)
    assert steps[-1].automatic and steps[-1].shape == (10,) and steps[-1].parameters == 64 * 10 + 10
    assert steps[0].parameters == 16 * 1 * 9 + 16 and steps[1].parameters == 32
    cifar = plan(starter_stack(), in_channels=3, input_size=(32, 32), num_classes=100)
    assert cifar[8].shape == (32 * 8 * 8,) and cifar[-1].shape == (100,)


def test_flatten_is_added_where_needed() -> None:
    layers = [{"type": "conv", "channels": 4, "kernel": 3, "padding": 0}, {"type": "linear", "features": 8}]
    steps = plan(layers, in_channels=1, input_size=(5, 5), num_classes=2)
    assert [s.kind for s in steps] == ["conv", "flatten", "linear", "linear"]
    assert steps[1].automatic and steps[1].shape == (36,) and steps[2].parameters == 36 * 8 + 8
    bare = plan([], in_channels=3, input_size=(4, 4), num_classes=5)
    assert [s.kind for s in bare] == ["flatten", "linear"] and bare[-1].parameters == 48 * 5 + 5


@pytest.mark.parametrize(("layers", "message"), [
    ([{"type": "lstm"}], "unknown type"),
    ([{"type": "conv", "size": 3}], "unknown setting"),
    ([{"type": "conv", "channels": 0}], "at least 1"),
    ([{"type": "conv", "channels": 2.5}], "whole number"),
    ([{"type": "conv", "kernel": 9, "padding": 0}], "does not fit"),
    ([{"type": "maxpool", "kernel": 2}] * 4, "smaller than"),
    ([{"type": "flatten"}, {"type": "conv"}], "needs image maps"),
    ([{"type": "dropout", "p": 1.0}], "between 0 and 1"),
])
def test_errors_name_the_layer(layers, message) -> None:
    with pytest.raises(LayerError, match=message):
        plan(layers, in_channels=1, input_size=(8, 8), num_classes=3)


def test_built_module_matches_the_plan_and_the_code() -> None:
    torch = pytest.importorskip("torch")
    from labs.deep_learning.model import build_layer_stack

    layers = [*starter_stack()[:4], {"type": "avgpool", "size": 2}, {"type": "linear", "features": 12}]
    steps = plan(layers, in_channels=3, input_size=(32, 32), num_classes=7)
    model = build_layer_stack(num_classes=7, in_channels=3, input_size=(32, 32), layers=layers)
    assert sum(p.numel() for p in model.parameters()) == sum(s.parameters for s in steps)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 7)
    namespace: dict = {}
    exec(pytorch_code(steps, in_channels=3), namespace)  # noqa: S102 - our own generated code
    written = namespace["LayerStack"]()
    assert sum(p.numel() for p in written.parameters()) == sum(s.parameters for s in steps)
    assert written(torch.zeros(1, 3, 32, 32)).shape == (1, 7)
    with pytest.raises(ValueError, match="no pretrained"):
        build_layer_stack(num_classes=2, in_channels=1, input_size=(8, 8), pretrained="imagenet")
    with pytest.raises(LayerError):
        build_layer_stack(num_classes=2, in_channels=1, input_size=(8, 8), layers=[{"type": "maxpool",
                                                                                      "kernel": 16}])
