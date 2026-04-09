import dataclasses

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, rand_strided


@dataclasses.dataclass(frozen=True)
class BinarySpec:
    name: str
    ref: callable
    lhs_range: tuple[float, float]
    rhs_range: tuple[float, float]
    tolerances: dict[torch.dtype, tuple[float, float]]


_DEFAULT_TOLERANCES = {
    torch.float32: (1e-6, 1e-6),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}

_POW_TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (3e-2, 3e-2),
}

_BINARY_SPECS = (
    BinarySpec("div", torch.div, (-5.0, 5.0), (0.1, 5.0), _DEFAULT_TOLERANCES),
    BinarySpec(
        "max", torch.maximum, (-5.0, 5.0), (-5.0, 5.0), _DEFAULT_TOLERANCES
    ),
    BinarySpec(
        "min", torch.minimum, (-5.0, 5.0), (-5.0, 5.0), _DEFAULT_TOLERANCES
    ),
    BinarySpec(
        "mod", torch.remainder, (-5.0, 5.0), (0.5, 4.0), _DEFAULT_TOLERANCES
    ),
    BinarySpec("pow", torch.pow, (0.1, 5.0), (0.1, 3.0), _POW_TOLERANCES),
)


def _uniform_input(shape, strides, *, low, high, dtype, device):
    input = rand_strided(shape, strides, dtype=dtype, device=device)
    input.mul_(high - low).add_(low)
    return input


def _resolve_tolerance(spec, dtype, default_rtol, default_atol):
    return spec.tolerances.get(dtype, (default_rtol, default_atol))


def _call_binary(spec, input, other, out):
    getattr(infini.ops, spec.name)(input, other, out)
    return out


def _torch_binary(spec, input, other, out):
    try:
        result = spec.ref(input, other)
    except RuntimeError as exc:
        pytest.skip(f"`torch.{spec.name}` does not support this case: {exc}")

    out.copy_(result.to(out.dtype))
    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, other_strides, out_strides",
    (
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None, None),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
        ((13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
        ((13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ),
)
@pytest.mark.parametrize("spec", _BINARY_SPECS, ids=lambda spec: spec.name)
def test_binary_elementwise(
    spec, shape, input_strides, other_strides, out_strides, dtype, device, rtol, atol
):
    input = _uniform_input(
        shape,
        input_strides,
        low=spec.lhs_range[0],
        high=spec.lhs_range[1],
        dtype=dtype,
        device=device,
    )
    other = _uniform_input(
        shape,
        other_strides,
        low=spec.rhs_range[0],
        high=spec.rhs_range[1],
        dtype=dtype,
        device=device,
    )
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    resolved_rtol, resolved_atol = _resolve_tolerance(spec, dtype, rtol, atol)

    return Payload(
        _call_binary,
        _torch_binary,
        (spec, input, other, out),
        {},
        rtol=resolved_rtol,
        atol=resolved_atol,
    )
