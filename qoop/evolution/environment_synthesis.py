
from dataclasses import dataclass, field
from .environment_parent import Metadata

@dataclass
class MetadataSynthesis(Metadata):
    num_cnot: int = field(default_factory=lambda: "num_cnot")
    depth: int = field(default_factory=lambda: "depth")
    num_rx: int = field(default_factory=lambda: 0)  # Add count for RX gates
    num_ry: int = field(default_factory=lambda: 0)  # Add count for RY gates
    num_rz: int = field(default_factory=lambda: 0)  # Add count for RZ gates