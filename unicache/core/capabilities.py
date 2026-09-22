"""Static and runtime capability contracts for UniCache operators."""

from __future__ import annotations

from dataclasses import dataclass


WILDCARD_CAPABILITY = "*"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Describe where an operator can run and what it does.

    Static fields are embedded in the execution plan. Runtime adapters use the
    same descriptor to reject unsupported devices or physical cache layouts
    before invoking an operator.
    """

    stages: frozenset[str] = frozenset({WILDCARD_CAPABILITY})
    phases: frozenset[str] = frozenset({WILDCARD_CAPABILITY})
    devices: frozenset[str] = frozenset({WILDCARD_CAPABILITY})
    layouts: frozenset[str] = frozenset({WILDCARD_CAPABILITY})
    requires: frozenset[str] = frozenset()
    produces_selection: bool = False
    stateful: bool = False
    mutation_scope: str = "none"
    physical_storage_mutation: bool = False
    accepts_quantized_input: bool = False

    def validate_static(self, *, operator: str, stage: str) -> None:
        if not self._supports(self.stages, stage):
            raise ValueError(
                f"Operator {operator} does not support stage {stage}; "
                f"supported stages are {sorted(self.stages)}"
            )

    def validate_runtime(
        self,
        *,
        operator: str,
        stage: str,
        phase: str,
        device: str,
        layout: str,
        available_features: set[str],
    ) -> None:
        self.validate_static(operator=operator, stage=stage)
        for label, supported, value in (
            ("phase", self.phases, phase),
            ("device", self.devices, device),
            ("cache layout", self.layouts, layout),
        ):
            if not self._supports(supported, value):
                raise ValueError(
                    f"Operator {operator} does not support {label} {value}; "
                    f"supported values are {sorted(supported)}"
                )
        missing = sorted(self.requires - available_features)
        if missing:
            raise ValueError(f"Operator {operator} requires unavailable runtime features: {missing}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stages": sorted(self.stages),
            "phases": sorted(self.phases),
            "devices": sorted(self.devices),
            "layouts": sorted(self.layouts),
            "requires": sorted(self.requires),
            "produces_selection": self.produces_selection,
            "stateful": self.stateful,
            "mutation_scope": self.mutation_scope,
            "physical_storage_mutation": self.physical_storage_mutation,
            "accepts_quantized_input": self.accepts_quantized_input,
        }

    @staticmethod
    def _supports(supported: frozenset[str], value: str) -> bool:
        return WILDCARD_CAPABILITY in supported or value in supported
