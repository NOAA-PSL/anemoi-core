# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.layers.activations import leaky_hardtanh


class BaseBounding(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    This class defines an interface for bounding strategies which are used to apply a specific
    restriction to the predictions of a model.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        """Initializes the bounding strategy.
        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict, optional
            A dictionary containing the statistics of the variables.
        name_to_index_stats : dict, optional
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
        """
        super().__init__()
        for var in variables:
            if var not in name_to_index:
                raise KeyError(
                    f"{self.__class__.__name__}: variable '{var}' is not present in the name_to_index mapping."
                )

        self.name_to_index = name_to_index
        self.variables = variables
        self.data_index = self._create_index(variables=self.variables)
        self.statistics = statistics
        self.name_to_index_stats = name_to_index_stats

    def _create_index(self, variables: list[str]) -> InputTensorIndex:
        return torch.tensor([i for name, i in self.name_to_index.items() if name in variables], dtype=torch.int)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the bounding to the predictions.

        Parameters
        ----------
        x : torch.Tensor
            The tensor containing the predictions that will be bounded.

        Returns
        -------
        torch.Tensor
        A tensor with the bounding applied.
        """
        pass


class ReluBounding(BaseBounding):
    """Initializes the bounding with a ReLU activation / zero clamping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.relu(x[..., self.data_index])
        return x


class LeakyReluBounding(BaseBounding):
    """Initializes the bounding with a Leaky ReLU activation / zero clamping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.leaky_relu(x[..., self.data_index])
        return x


class NormalizedReluBounding(BaseBounding):
    """Bounding variable with a ReLU activation and customizable normalized thresholds."""

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: list[float],
        normalizer: list[str],
        statistics: dict,
        name_to_index_stats: dict,
    ) -> None:
        """Initializes the NormalizedReluBounding with the specified parameters.

        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict
            A dictionary containing the statistics of the variables (mean, std, min, max, etc.).
        min_val : list[float]
            The minimum values for the ReLU activation. It should be given in the same order as the variables.
        normalizer : list[str]
            A list of normalization types to apply, one per variable. Options: 'mean-std', 'min-max', 'max', 'std'.
        name_to_index_stats : dict
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
        """
        if len(normalizer) != len(variables):
            raise ValueError(
                "The length of the normalizer list must match the number of variables in NormalizedReluBounding."
            )
        if len(min_val) != len(variables):
            raise ValueError(
                "The length of the min_val list must match the number of variables in NormalizedReluBounding."
            )
        if not all(norm in {"mean-std", "min-max", "max", "std"} for norm in normalizer):
            raise ValueError(
                "Each normalizer must be one of: 'mean-std', 'min-max', 'max', 'std' in NormalizedReluBounding."
            )

        super().__init__(
            variables=variables,
            name_to_index=name_to_index,
            statistics=statistics,
            name_to_index_stats=name_to_index_stats,
        )

        # Silently skip variables absent from this dataset (matches BaseBounding._create_index).
        kept = [(ii, var) for ii, var in enumerate(variables) if var in name_to_index]
        self.variables = [var for _, var in kept]
        self.min_val = [min_val[ii] for ii, _ in kept]
        self.normalizer = [normalizer[ii] for ii, _ in kept]

        # Create data index for the variables to be bounded in order from configuration
        self.data_index = torch.tensor([name_to_index[var] for var in self.variables], dtype=self.data_index.dtype)
        # Compute normalized min values
        norm_min_val = torch.zeros(len(self.variables), dtype=torch.float32)
        for ii, variable in enumerate(self.variables):
            stat_index = self.name_to_index_stats[variable]
            if self.normalizer[ii] == "mean-std":
                mean = self.statistics["mean"][stat_index]
                std = self.statistics["stdev"][stat_index]
                norm_min_val[ii] = (self.min_val[ii] - mean) / std
            elif self.normalizer[ii] == "min-max":
                min_stat = self.statistics["min"][stat_index]
                max_stat = self.statistics["max"][stat_index]
                norm_min_val[ii] = (self.min_val[ii] - min_stat) / (max_stat - min_stat)
            elif self.normalizer[ii] == "max":
                max_stat = self.statistics["max"][stat_index]
                norm_min_val[ii] = self.min_val[ii] / max_stat
            elif self.normalizer[ii] == "std":
                std = self.statistics["stdev"][stat_index]
                norm_min_val[ii] = self.min_val[ii] / std
        # register the normalized min values as a buffer to ensure they are moved to the correct device
        self.register_buffer("norm_min_val", norm_min_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ReLU activation with the normalized minimum values to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to process.

        Returns
        -------
        torch.Tensor
            The processed tensor with bounding applied.
        """
        x[..., self.data_index] = (
            torch.nn.functional.relu(x[..., self.data_index] - self.norm_min_val) + self.norm_min_val
        )
        return x


class NormalizedLeakyReluBounding(NormalizedReluBounding):
    """Initializes the bounding with a Leaky ReLU activation and customizable normalized thresholds."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = (
            torch.nn.functional.leaky_relu(x[..., self.data_index] - self.norm_min_val) + self.norm_min_val
        )
        return x


class HardtanhBounding(BaseBounding):
    """Initializes the bounding with specified minimum and maximum values for bounding.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    statistics : dict, optional
        A dictionary containing the statistics of the variables.
    name_to_index_stats : dict, optional
        A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        return x


class LeakyHardtanhBounding(HardtanhBounding):
    """Initializes the bounding with a Leaky HardTanh activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = leaky_hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)
        return x


class FractionBounding(BaseBounding):
    """Initializes the FractionBounding with specified parameters.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    total_var : str
        A string representing a variable from which a secondary variable is derived. For
        example, in the case of convective precipitation (Cp), total_var = Tp (total precipitation).
    statistics : dict, optional
        A dictionary containing the statistics of the variables.
    name_to_index_stats : dict, optional
        A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        total_var: str,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val
        self.total_variable = self._create_index(variables=[total_var])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the HardTanh bounding  to the data_index variables
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x


class LeakyFractionBounding(FractionBounding):
    """Initializes the bounding with a Leaky HardTanh activation and a fraction of the total variable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the LeakyHardTanh bounding  to the data_index variables
        x[..., self.data_index] = leaky_hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x


class GlobalDryMassFixer(BaseBounding):
    """Conserves global dry air mass (as in ACE): keeps the area-weighted global mean of the dry surface
    pressure, sp - g * tcw, equal to its value in the latest input state by adding a spatially uniform
    correction to sp.

    Works on the normalized tensors. sp and the water column must be normalized with a scale of stdev
    ("mean-std" or "std"); the offsets cancel in the difference. Needs the latest input state (``x_ref``,
    the residual skip) and the data-node area weights (``bind``); the grid must not be sharded.

    Parameters
    ----------
    variables : list[str]
        The surface pressure variable, e.g. ``[sp]``. It receives the correction.
    name_to_index : dict
        Model output name to index.
    water_var : str
        Total column water (kg m-2), e.g. ``tcw``.
    gravity : float
        m s-2; converts the water column to Pa.
    statistics, name_to_index_stats : dict
        Dataset statistics (stdev) and their name to index mapping.
    area_attr : str
        Data-node attribute in the graph with the area weights.
    """

    needs_input = True

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        water_var: str = "tcw",
        gravity: float = 9.80665,
        statistics: dict,
        name_to_index_stats: dict,
        area_attr: str = "area_weight",
    ) -> None:
        super().__init__(
            variables=variables,
            name_to_index=name_to_index,
            statistics=statistics,
            name_to_index_stats=name_to_index_stats,
        )
        if len(variables) != 1 or water_var not in name_to_index:
            raise ValueError(f"GlobalDryMassFixer: needs one pressure variable and {water_var} in the model output.")
        self.pressure_var, self.water_var, self.area_attr = variables[0], water_var, area_attr
        self.sp_out, self.w_out = name_to_index[self.pressure_var], name_to_index[water_var]
        stdev = statistics["stdev"]
        # dry-pressure change per unit of normalized water, in units of normalized sp
        self.ratio = float(
            gravity * stdev[name_to_index_stats[water_var]] / stdev[name_to_index_stats[self.pressure_var]]
        )
        self.sp_in = self.w_in = None
        self.register_buffer("area", torch.empty(0), persistent=False)

    def bind(self, node_store, input_name_to_index: dict) -> None:
        """Set the area weights (normalized to sum 1) and the input indices of sp and the water column."""
        area = torch.as_tensor(node_store[self.area_attr]).reshape(-1).to(torch.float64)
        self.area = (area / area.sum()).to(device=self.area.device, dtype=torch.float32)
        self.sp_in, self.w_in = input_name_to_index[self.pressure_var], input_name_to_index[self.water_var]

    def forward(self, x: torch.Tensor, x_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, time, ensemble, grid, vars); x_ref: the latest input state, same layout, input variables
        if x_ref is None or self.sp_in is None:
            raise RuntimeError("GlobalDryMassFixer needs the latest input state, and bind() to have been called.")
        if x.shape[-2] != self.area.numel():
            raise RuntimeError(
                f"GlobalDryMassFixer: {x.shape[-2]} grid points vs {self.area.numel()} area weights (sharded grid?)."
            )
        area = self.area.to(x.dtype)
        d_sp = (x_ref[..., self.sp_in] - x[..., self.sp_out]) @ area
        d_w = (x_ref[..., self.w_in] - x[..., self.w_out]) @ area
        x = x.clone()  # the global means above depend on x
        x[..., self.sp_out] += (d_sp - self.ratio * d_w).unsqueeze(-1)
        return x


def _build_dataset_boundings(
    bounding_cfgs: Any,
    data_indices: IndexCollection,
    statistics: dict | None,
) -> nn.ModuleList:
    """Build the list of model-output bounding modules from configuration.

    This is a thin factory over Hydra's ``instantiate`` that reads the iterable
    ``model_config.model.bounding`` and instantiates each entry while injecting
    the common keyword arguments required by bounding modules:
    ``name_to_index``, ``statistics``, and ``name_to_index_stats``. The result
    is returned as an ``nn.ModuleList`` preserving the order of the config.

    Parameters
    ----------
    model_config : Any
        Object with a ``model`` attribute containing an iterable ``bounding``
        (e.g. a list of Hydra configs). If absent or empty, an empty
        ``nn.ModuleList`` is returned.
    data_indices : IndexCollection
        Object providing the mappings:
        ``data_indices.model.output.name_to_index`` and
        ``data_indices.data.input.name_to_index``. These are forwarded to each
        instantiated bounding module as ``name_to_index`` and
        ``name_to_index_stats`` respectively.
    statistics : dict[str, dict] | None
        Optional dataset/model statistics passed to each bounding module. Use
        ``None`` if not required by the configured classes.

    Returns
    -------
    torch.nn.ModuleList
        The instantiated bounding modules, in the same order as specified in
        ``model_config.model.bounding``. May be empty.
    """

    return nn.ModuleList(
        [
            instantiate(
                cfg,
                name_to_index=data_indices.model.output.name_to_index,
                statistics=statistics,
                name_to_index_stats=data_indices.data.input.name_to_index,
            )
            for cfg in bounding_cfgs
        ]
    )


def build_boundings(
    boundings_config: dict[str, list],
    data_indices: dict[str, IndexCollection],
    statistics: dict[str, dict] | None,
) -> nn.ModuleDict:
    """Build the model-output bounding modules from configuration.

    This is a thin factory that creates a ``nn.ModuleDict`` of bounding
    modules by invoking ``_build_dataset_boundings`` for each dataset
    specified in ``data_indices``.

    Parameters
    ----------
    boundings_config : dict[str, list]
        Dictionary containing iterables ``bounding`` (e.g. a list of Hydra configs).
        If absent or empty, an empty ``nn.ModuleDict`` is returned.
    data_indices : dict[str, IndexCollection]
        Dictionary mapping dataset names to data indices objects. Each
        data indices object must provide the mappings:
        ``data_indices.model.output.name_to_index`` and
        ``data_indices.data.input.name_to_index``. These are forwarded to each
        instantiated bounding module as ``name_to_index`` and
        ``name_to_index_stats`` respectively.
    statistics : dict[str, dict] | None
        Dictionary mapping dataset names to optional dataset/model statistics
        passed to each bounding module. Use ``None`` if not required by the
        configured classes.
    """
    bounding_modules = nn.ModuleDict()
    for dataset_name in data_indices.keys():
        bounding_modules[dataset_name] = _build_dataset_boundings(
            boundings_config.get(dataset_name, []),
            data_indices=data_indices[dataset_name],
            statistics=statistics[dataset_name],
        )

    return bounding_modules
