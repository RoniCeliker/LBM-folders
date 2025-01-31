from enum import IntEnum


class BoundaryType(IntEnum):
    ORPHAN = 0
    PERIODIC = 1
    BOUNCE_BACK = 2
    ANTI_BOUNCE_BACK = 3
    SYMMETRY = 10


class BoundaryClass(IntEnum):
    NONE = -1
    WALL = 0
    INTERNAL = 1
    EXTERNAL = 2
