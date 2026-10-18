# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.bounding import GlobalDryMassFixer

G = 9.80665
# statistics order: sp, tcw, t, forcing
STATS = {"mean": [98000.0, 25.0, 270.0, 0.5], "stdev": [1200.0, 16.0, 20.0, 0.3]}
STATS_INDEX = {"sp": 0, "tcw": 1, "t": 2, "f": 3}
OUTPUT_INDEX = {"tcw": 0, "sp": 1, "t": 2}
INPUT_INDEX = {"sp": 0, "tcw": 1, "t": 2, "f": 3}


def physical_dry_mass(x_norm: torch.Tensor, sp_i: int, w_i: int, area: torch.Tensor) -> torch.Tensor:
    sp = x_norm[..., sp_i] * STATS["stdev"][0] + STATS["mean"][0]
    tcw = x_norm[..., w_i] * STATS["stdev"][1]  # "std" normalized
    return (sp - G * tcw) @ (area / area.sum())


@pytest.fixture()
def fixer():
    torch.manual_seed(0)
    area = torch.rand(50, dtype=torch.float64) + 0.5
    fixer = GlobalDryMassFixer(
        variables=["sp"], name_to_index=OUTPUT_INDEX, water_var="tcw", statistics=STATS, name_to_index_stats=STATS_INDEX
    )
    fixer.bind({"area_weight": area.unsqueeze(-1)}, INPUT_INDEX)
    return fixer, area


def test_conserves_dry_mass(fixer) -> None:
    fixer, area = fixer
    x_ref = torch.randn(2, 1, 1, 50, 4, dtype=torch.float64)
    x = torch.randn(2, 1, 1, 50, 3, dtype=torch.float64)
    out = fixer(x.clone(), x_ref=x_ref)
    area = fixer.area.double()
    torch.testing.assert_close(physical_dry_mass(out, 1, 0, area), physical_dry_mass(x_ref, 0, 1, area))
    # only sp changes, by a spatially uniform amount
    torch.testing.assert_close(out[..., [0, 2]], x[..., [0, 2]])
    shift = out[..., 1] - x[..., 1]
    torch.testing.assert_close(shift, shift[..., :1].expand_as(shift))


def test_differentiable(fixer) -> None:
    fixer, _ = fixer
    x_ref = torch.randn(1, 1, 1, 50, 4, dtype=torch.float64, requires_grad=True)
    x = torch.randn(1, 1, 1, 50, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: fixer(a * 1.0, x_ref=b), (x, x_ref))


def test_needs_input_and_full_grid(fixer) -> None:
    fixer, _ = fixer
    with pytest.raises(RuntimeError):
        fixer(torch.randn(1, 1, 1, 50, 3))
    with pytest.raises(RuntimeError):
        fixer(torch.randn(1, 1, 1, 25, 3), x_ref=torch.randn(1, 1, 1, 25, 4))
