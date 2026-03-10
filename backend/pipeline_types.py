"""
Pipeline domain model: types, schemas, and data structures for the
image analysis recipe system.
"""
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import uuid


class DataType(str, Enum):
    """Data types that flow between pipeline steps."""
    IMAGE = "IMAGE"           # BGR uint8 numpy array (H×W×3)
    GRAYSCALE = "GRAYSCALE"   # Single-channel uint8 array (H×W)
    MASK = "MASK"             # Binary mask uint8 (0 or 255) (H×W)
    SCALAR = "SCALAR"         # Single numeric value
    HISTOGRAM = "HISTOGRAM"   # Histogram array(s)
    CONTOURS = "CONTOURS"     # List of OpenCV contours


# Auto-conversion matrix: (from_type, to_type) -> conversion function name
# None means identity; missing key means incompatible.
AUTO_CONVERSIONS = {
    (DataType.IMAGE, DataType.GRAYSCALE): "bgr_to_gray",
    (DataType.IMAGE, DataType.IMAGE): None,
    (DataType.GRAYSCALE, DataType.IMAGE): "gray_to_bgr",
    (DataType.GRAYSCALE, DataType.GRAYSCALE): None,
    (DataType.GRAYSCALE, DataType.MASK): None,
    (DataType.MASK, DataType.IMAGE): "gray_to_bgr",
    (DataType.MASK, DataType.GRAYSCALE): None,
    (DataType.MASK, DataType.MASK): None,
}


def can_convert(from_type: DataType, to_type: DataType) -> bool:
    """Check if an auto-conversion exists between two data types."""
    if from_type == to_type:
        return True
    return (from_type, to_type) in AUTO_CONVERSIONS


@dataclass
class ParamSchema:
    """Describes one editable parameter of a pipeline step."""
    name: str
    label: str
    type: str       # "int", "float", "bool", "enum", "string", "file_path"
    default: Any
    required: bool = True
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None   # For enum type
    description: Optional[str] = None
    odd_only: Optional[bool] = None       # For kernel sizes

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class StepDefinition:
    """Template definition of a pipeline step type (immutable catalog entry)."""
    id: str
    name: str
    category: str          # "io", "adjustment", "filter", "analysis", "detection"
    description: str
    icon: str              # Material icon name
    input_type: DataType
    output_type: DataType
    params: List[ParamSchema] = field(default_factory=list)
    side_output_types: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "icon": self.icon,
            "input_type": self.input_type.value,
            "output_type": self.output_type.value,
            "params": [p.to_dict() for p in self.params],
            "side_output_types": self.side_output_types,
        }


@dataclass
class StepInstance:
    """A user-configured step placed in a pipeline."""
    instance_id: str
    step_def_id: str
    param_values: Dict[str, Any] = field(default_factory=dict)
    order: int = 0

    @staticmethod
    def create(step_def_id: str, param_values: Optional[Dict] = None, order: int = 0):
        return StepInstance(
            instance_id=str(uuid.uuid4()),
            step_def_id=step_def_id,
            param_values=param_values or {},
            order=order,
        )

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "step_def_id": self.step_def_id,
            "param_values": self.param_values,
            "order": self.order,
        }

    @staticmethod
    def from_dict(d: dict) -> "StepInstance":
        return StepInstance(
            instance_id=d.get("instance_id", str(uuid.uuid4())),
            step_def_id=d["step_def_id"],
            param_values=d.get("param_values", {}),
            order=d.get("order", 0),
        )


@dataclass
class PipelineDocument:
    """Full recipe/pipeline document for persistence."""
    schema_version: int = 1
    name: str = ""
    description: str = ""
    steps: List[StepInstance] = field(default_factory=list)
    created_at: str = ""
    modified_at: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "PipelineDocument":
        steps = [StepInstance.from_dict(s) for s in d.get("steps", [])]
        return PipelineDocument(
            schema_version=d.get("schema_version", 1),
            name=d.get("name", ""),
            description=d.get("description", ""),
            steps=steps,
            created_at=d.get("created_at", ""),
            modified_at=d.get("modified_at", ""),
        )


@dataclass
class StepResult:
    """Result of executing one pipeline step."""
    success: bool
    primary_output: Any = None       # numpy array (image) or None
    side_outputs: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    output_type: Optional[DataType] = None


@dataclass
class StepError:
    """Error from a failed pipeline step."""
    step_index: int
    step_def_id: str
    error_code: str
    message: str
    param_name: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "step_index": self.step_index,
            "step_def_id": self.step_def_id,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.param_name:
            d["param_name"] = self.param_name
        return d


@dataclass
class PipelineResult:
    """Result of executing an entire pipeline (or partial up to a step)."""
    success: bool
    step_results: List[StepResult] = field(default_factory=list)
    errors: List[StepError] = field(default_factory=list)
    executed_up_to: int = -1
    data: Any = None  # Final data dict from the pipeline (multi-image)
