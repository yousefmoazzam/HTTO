from enum import IntEnum, unique


@unique
class PipelineStages(IntEnum):
    """An enumeration of available pipeline stages."""

    FILTER = 0
    NORMALIZE = 1
    STRIPES = 2
    CENTER = 3
    RECONSTRUCT = 4
