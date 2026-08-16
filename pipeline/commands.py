"""A ``Step`` is one unit of work run inside a specific conda environment.

Every stage is expressed as a small block of bash that (1) activates a conda
env, (2) exports any extra environment variables, (3) ``cd``s into the owning
repository, and (4) runs a command.  The same ``Step`` is rendered either into
a ``bash -c`` invocation (local executor) or into an sbatch script (SLURM
executor), so there is a single source of truth for how each tool is called.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field


@dataclass
class Step:
    name: str                       # unique, filesystem-safe, e.g. "align_tissue1"
    env: str                        # conda environment name
    cwd: str                        # working directory (owning repo root)
    invocation: str                 # the command line to run (already shell-quoted)
    exports: list[str] = field(default_factory=list)   # raw "KEY=VALUE" assignments
    pre_lines: list[str] = field(default_factory=list)  # bash lines run before invocation
    stage: str = ""                 # logical stage name (stack/align/tile/segment/stitch)


def py(argv: list[str]) -> str:
    """Shell-quote a fully-static python invocation."""
    return shlex.join(argv)


def render_body(step: Step, include_conda_hook: bool = True) -> str:
    """Render the portable bash body shared by both executors."""
    # Note: intentionally no `-u`; some conda activation scripts reference
    # unset variables and would abort under `set -u`.
    lines = ["set -eo pipefail", ""]
    if include_conda_hook:
        lines.append('eval "$(conda shell.bash hook)"')
    lines.append(f"conda activate {shlex.quote(step.env)}")
    for assignment in step.exports:
        lines.append(f"export {assignment}")
    lines.append("")
    lines.append(f"cd {shlex.quote(step.cwd)}")
    if step.pre_lines:
        lines.append("")
        lines.extend(step.pre_lines)
    lines.append("")
    lines.append(step.invocation)
    return "\n".join(lines) + "\n"
