"""
CSCS-AL (partial-correction release)
=====================================
Dataset-aware curriculum active learning with targeted partial annotation
for 3D medical image segmentation.

This package covers the full iterative method: dataset-aware cold-start,
DCR-governed adaptive acquisition, entropy-ranked partial correction, and
provenance-aware training weights. It is the code release accompanying the
"Dataset-Aware Curriculum Active Learning with Targeted Partial Annotation
for 3D Medical Image Segmentation" submission.

Note: this is a distinct code release from the earlier cold-start-only
CSCS package (initial-seed selection alone, no iterative rounds or partial
correction). If you have both installed, import from the submodules
explicitly to avoid ambiguity.
"""

__version__ = "0.1.0"
