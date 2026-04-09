import dataclasses

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, rand_strided


@dataclasses.dataclass(frozen=True)
class UnarySpec:
    name: str
    ref: callable
    low: float
    high: float
    tolerances: dict[torch.dtype, tuple[float, float]]


_DEFAULT_TOLERANCES = {
    torch.float32: (1e-6, 1e-6),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}

_TRIG_TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (3e-2, 3e-2),
}

_UNARY_SPECS = (
    UnarySpec("abs", torch.abs, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("acos", torch.acos, -0.95, 0.95, _TRIG_TOLERANCES),
    UnarySpec("acosh", torch.acosh, 1.05, 4.0, _DEFAULT_TOLERANCES),
    UnarySpec("asin", torch.asin, -0.95, 0.95, _TRIG_TOLERANCES),
    UnarySpec("asinh", torch.asinh, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("atan", torch.atan, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("atanh", torch.atanh, -0.8, 0.8, _DEFAULT_TOLERANCES),
    UnarySpec("ceil", torch.ceil, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("cos", torch.cos, -6.0, 6.0, _TRIG_TOLERANCES),
    UnarySpec("cosh", torch.cosh, -3.0, 3.0, _DEFAULT_TOLERANCES),
    UnarySpec("erf", torch.erf, -3.0, 3.0, _DEFAULT_TOLERANCES),
    UnarySpec("floor", torch.floor, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("log", torch.log, 0.05, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("neg", torch.neg, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("reciprocal", torch.reciprocal, 0.05, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("round", torch.round, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("sign", torch.sign, -5.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("sinh", torch.sinh, -3.0, 3.0, _DEFAULT_TOLERANCES),
    UnarySpec("sqrt", torch.sqrt, 0.0, 5.0, _DEFAULT_TOLERANCES),
    UnarySpec("tan", torch.tan, -1.2, 1.2, _TRIG_TOLERANCES),
)


def _uniform_input(shape, strides, *, low, high, dtype, device):
    input = rand_strided(shape, strides, dtype=dtype, device=device)
    input.mul_(high - low).add_(low)
    return input


def _resolve_tolerance(spec, dtype, default_rtol, default_atol):
    return spec.tolerances.get(dtype, (default_rtol, default_atol))


def _call_unary(spec, input, out):
    getattr(infini.ops, spec.name)(input, out)
    return out


def _torch_unary(spec, input, out):
    try:
        result = spec.ref(input)
    except RuntimeError as exc:
        pytest.skip(f"`torch.{spec.name}` does not support this case: {exc}")

    out.copy_(result.to(out.dtype))
    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides",
    (
        ((13, 4), None, None),
        ((13, 4), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None),
        ((13, 4, 4), None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
        ((13, 16, 2), (128, 4, 1), (64, 4, 1)),
        ((4, 4, 5632), None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
    ),
)
@pytest.mark.parametrize("spec", _UNARY_SPECS, ids=lambda spec: spec.name)
def test_unary_elementwise(
    spec, shape, input_strides, out_strides, dtype, device, rtol, atol
):
    input = _uniform_input(
        shape,
        input_strides,
        low=spec.low,
        high=spec.high,
        dtype=dtype,
        device=device,
    )
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    resolved_rtol, resolved_atol = _resolve_tolerance(spec, dtype, rtol, atol)

    return Payload(
        _call_unary,
        _torch_unary,
        (spec, input, out),
        {},
        rtol=resolved_rtol,
        atol=resolved_atol,
    )
