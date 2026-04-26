"""Custom process models for the process chain benchmark.

Defines MixingProcess (extended) and HeatingProcess with physical
quantities using v1 opensemantic classes throughout.
"""

from __future__ import annotations

from pydantic.v1 import Field

from opensemantic.v1 import OswBaseModel
from opensemantic.core.v1 import Material, Process
from opensemantic.lab.v1 import MixingProcess as _BaseMixingProcess
from opensemantic.characteristics.quantitative.v1 import (
    Mass,
    RotationalFrequency,
    Temperature,
    Time,
)


class ProcessInput(OswBaseModel):
    """An input or output material with its mass."""

    class Config:
        schema_extra = {
            "title": "ProcessInput",
            "description": "A material with its mass used as process input or output.",
        }

    material: Material = Field(
        None,
        title="Material",
        description="Name or reference of the material.",
        range="Category:OSW31ca9a739cb24079b36824045c0832aa",
    )
    mass: Mass = Field(
        None,
        title="Mass",
    )


class MixingProcess(_BaseMixingProcess):
    """Mixing process with duration, rotational speed, and typed inputs/outputs."""

    class Config:
        schema_extra = {
            "uuid": "29355236-8bcf-4a36-8168-2671f8d47d8f",
            "title": "MixingProcess",
            "title*": {"en": "Mixing process", "de": "Mischprozess"},
            "description": (
                "A mixing process with duration, rotational speed, "
                "and typed material inputs/outputs."
            ),
        }

    type: list[str] | None = ["Category:OSW293552368bcf4a3681682671f8d47d8f"]
    duration: Time = Field(None, title="Duration")
    mixing_speed: RotationalFrequency = Field(None, title="Mixing speed")
    input_materials: list[ProcessInput] = Field(
        None, title="Input materials"
    )
    output_materials: list[ProcessInput] = Field(
        None, title="Output materials"
    )


ExtendedMixingProcess = MixingProcess


class HeatingProcess(Process):
    """Heating or curing process with temperature and duration."""

    class Config:
        schema_extra = {
            "uuid": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "title": "HeatingProcess",
            "title*": {"en": "Heating process", "de": "Heizprozess"},
            "description": (
                "A process involving heating or curing at a specified "
                "temperature for a given duration."
            ),
        }

    type: list[str] | None = ["Category:OSWb2c3d4e5f6a78901bcdef12345678901"]
    duration: Time = Field(None, title="Duration")
    temperature: Temperature = Field(None, title="Temperature")
    input_materials: list[ProcessInput] = Field(
        None, title="Input materials"
    )
    output_materials: list[ProcessInput] = Field(
        None, title="Output materials"
    )


# Paths to hide from the schema inventory — the original opensemantic
# mixing classes conflict with our extended MixingProcess and confuse the LLM.
HIDDEN_SCHEMA_PATHS = {
    "opensemantic.lab.v1.MixingProcess",
    "opensemantic.lab.v1.MixingOfChemicalSubstances",
    "opensemantic.lab.v1.MixingSourceProcess",
    "opensemantic.lab.v1.ChemicalSubstance",
    "opensemantic.lab.v1.ChemicalSubstanceType",
    "process_models._BaseMixingProcess",
    "process_models.ExtendedMixingProcess",
}


if __name__ == "__main__":
    import json

    import oold.static

    for cls in [ProcessInput, MixingProcess, HeatingProcess]:
        print(f"\n{'='*60}")
        print(f"Schema for {cls.__name__}:")
        print("=" * 60)
        schema = cls.export_schema(
            mode=oold.static.SchemaExportMode.PARTIAL,
            serialize="json",
        )
        print(json.dumps(json.loads(schema), indent=2))
