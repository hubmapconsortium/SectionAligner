"""SectionAligner pipeline — orchestrator for 3D tissue segmentation.

Runs a five-stage pipeline (stack → align → tile → segment → stitch) by
launching each stage as a subprocess in its own conda environment. Every stage's
tool lives in this repository (see :mod:`pipeline.paths`); the orchestrator
itself only needs the standard library plus PyYAML.
"""

__all__ = ["config", "layout", "paths", "commands", "stages", "executor", "orchestrator"]
