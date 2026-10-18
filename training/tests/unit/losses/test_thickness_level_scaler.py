# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers import ThicknessVariableLevelScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel


def scaling(names: list[str], floor: float = 0.1, total: float = 6.5) -> dict[str, float]:
    data_indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {n: i for i, n in enumerate(names)})
    scaler = ThicknessVariableLevelScaler(
        data_indices,
        group="pl",
        metadata_extractor=ExtractVariableGroupAndLevel(DictConfig({"default": "sfc", "pl": ["t", "q"]})),
        floor=floor,
        total=total,
    )
    values = scaler.get_scaling_values()
    return {n: float(values[i]) for n, i in data_indices.model.output.name_to_index.items()}


def test_thickness_weights() -> None:
    # levels 1, 100, 500, 1000 hPa: edges 0, 50.5, 300, 750, 1000 -> dp 50.5, 249.5, 450, 250 (mean 250)
    w = scaling(["2t", "t_500", "t_1", "t_1000", "t_100", "q_500", "q_1000"], floor=0.0, total=1.0)
    torch.testing.assert_close(
        torch.tensor([w["t_1"], w["t_100"], w["t_500"], w["t_1000"]]),
        torch.tensor([50.5, 249.5, 450.0, 250.0]) / 1000.0,
    )
    assert w["2t"] == 1.0
    # q has its own level set: edges 0, 750, 1000
    assert abs(w["q_500"] - 0.75) < 1e-6 and abs(w["q_1000"] - 0.25) < 1e-6


def test_floor_and_total() -> None:
    names = [f"t_{p}" for p in (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 700, 850, 1000)]
    w = scaling(names, floor=0.1, total=6.5)
    assert abs(sum(w.values()) - 6.5) < 1e-5
    # 1, 2 and 3 hPa are thinner than 0.1 of the mean layer: all three sit at the floor
    assert w["t_1"] == w["t_2"] == w["t_3"] > w["t_1000"] * 1.5 / 75.0
