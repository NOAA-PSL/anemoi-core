# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.normalizer import InputNormalizer


def test_stdev_floor() -> None:
    config = DictConfig(
        {
            "normalizer": {"default": "mean-std", "std": ["q_1", "q_500"], "stdev_min": {"q_*": 1.0e-5}},
            "forcing": [],
            "diagnostic": [],
        }
    )
    stdev = np.array([2.0e-7, 3.0e-3, 0.5])
    statistics = {
        "mean": np.array([3.0e-6, 2.0e-3, 270.0]),
        "stdev": stdev,
        "minimum": np.zeros(3),
        "maximum": np.ones(3),
    }
    data_indices = IndexCollection(data_config=config, name_to_index={"q_1": 0, "q_500": 1, "t_1": 2})
    normalizer = InputNormalizer(config=config.normalizer, data_indices=data_indices, statistics=statistics)
    x = torch.tensor([[1.0e-5, 1.0e-3, 271.0]])
    torch.testing.assert_close(normalizer.transform(x, in_place=False), torch.tensor([[1.0, 1.0 / 3.0, 2.0]]))
    assert stdev[0] == 2.0e-7  # the shared statistics are not modified
