from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpochMetrics:
    epoch: int
    train_box_loss: float = 0.0
    train_cls_loss: float = 0.0
    train_dfl_loss: float = 0.0
    val_box_loss: float = 0.0
    val_cls_loss: float = 0.0
    val_dfl_loss: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0

    @property
    def train_loss_total(self) -> float:
        return self.train_box_loss + self.train_cls_loss + self.train_dfl_loss

    @property
    def val_loss_total(self) -> float:
        return self.val_box_loss + self.val_cls_loss + self.val_dfl_loss


@dataclass
class ClassMetrics:
    class_name: str
    ap50: float = 0.0
    ap50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    num_instances: int = 0
    fp_background: int = 0
    fp_class_confusion: int = 0
    fn_missed: int = 0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom > 0 else 0.0


@dataclass
class SizeMetrics:
    small_map50: float = 0.0
    medium_map50: float = 0.0
    large_map50: float = 0.0
    small_count: int = 0
    medium_count: int = 0
    large_count: int = 0


@dataclass
class AnchorMetrics:
    total_objects: int = 0
    assigned_objects: int = 0
    unassigned_objects: int = 0

    @property
    def assignment_ratio(self) -> float:
        return self.assigned_objects / self.total_objects if self.total_objects > 0 else 1.0


@dataclass
class AnalysisReport:
    run_id: str
    epoch: int
    class_metrics: list[ClassMetrics] = field(default_factory=list)
    size_metrics: SizeMetrics = field(default_factory=SizeMetrics)
    anchor_metrics: AnchorMetrics = field(default_factory=AnchorMetrics)
    confusion_matrix: Optional[np.ndarray] = None
    class_names: list[str] = field(default_factory=list)
    weaknesses: list[dict] = field(default_factory=list)
    overall_map50: float = 0.0
    overall_map50_95: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "epoch": self.epoch,
            "overall_map50": self.overall_map50,
            "overall_map50_95": self.overall_map50_95,
            "weaknesses": self.weaknesses,
            "class_metrics": [
                {
                    "class": m.class_name,
                    "ap50": m.ap50,
                    "ap50_95": m.ap50_95,
                    "precision": m.precision,
                    "recall": m.recall,
                    "num_instances": m.num_instances,
                    "fp_background": m.fp_background,
                    "fp_class_confusion": m.fp_class_confusion,
                    "fn_missed": m.fn_missed,
                    "f1": m.f1,
                }
                for m in self.class_metrics
            ],
            "size_metrics": {
                "small_map50": self.size_metrics.small_map50,
                "medium_map50": self.size_metrics.medium_map50,
                "large_map50": self.size_metrics.large_map50,
                "small_count": self.size_metrics.small_count,
                "medium_count": self.size_metrics.medium_count,
                "large_count": self.size_metrics.large_count,
            },
            "anchor_metrics": {
                "assignment_ratio": self.anchor_metrics.assignment_ratio,
                "unassigned_objects": self.anchor_metrics.unassigned_objects,
            },
        }


@dataclass
class ConfigDelta:
    changes: dict = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0
    expected_map_improvement: float = 0.0
    estimated_epochs: int = 100
    priority: int = 1

    def apply_to(self, base_config: dict) -> dict:
        import copy
        new_config = copy.deepcopy(base_config)
        new_config["training"].update(self.changes)
        return new_config

    def to_dict(self) -> dict:
        return {
            "changes": self.changes,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "expected_map_improvement": self.expected_map_improvement,
            "estimated_epochs": self.estimated_epochs,
            "priority": self.priority,
        }


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def text_to_feature_vector(text: str, dim: int = 128) -> np.ndarray:
    """Lightweight deterministic text embedding via character n-gram hashing."""
    vec = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    for n in (2, 3, 4):
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            h = hash(gram) % dim
            vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
