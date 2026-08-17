"""Two ways to run a :class:`~pipeline.commands.Step`.

* :class:`LocalExecutor` runs it now, in a subprocess, streaming output to the
  console and to a per-step log file. Stages run sequentially on one machine.
* :class:`SlurmExecutor` renders it to an sbatch script and submits it,
  returning a job id so callers can chain stages with ``afterok`` dependencies.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from .commands import Step, render_body


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------
class LocalExecutor:
    def __init__(self, logs_dir: str, dry_run: bool = False) -> None:
        self.logs_dir = logs_dir
        self.dry_run = dry_run

    def run(self, step: Step, extra_env: dict | None = None) -> None:
        body = render_body(step)
        log_path = os.path.join(self.logs_dir, f"{step.name}.log")

        header = (
            f"\n{'=' * 72}\n"
            f"[{step.stage}] {step.name}   (conda env: {step.env})\n"
            f"    cwd: {step.cwd}\n"
            f"    log: {log_path}\n"
            f"{'=' * 72}"
        )
        print(header, flush=True)

        if self.dry_run:
            print(_indent(body))
            return

        os.makedirs(self.logs_dir, exist_ok=True)
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        with open(log_path, "w") as log_file:
            log_file.write(body + "\n" + ("-" * 72) + "\n")
            log_file.flush()
            proc = subprocess.Popen(
                ["bash", "-c", body],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
            returncode = proc.wait()

        if returncode != 0:
            raise RuntimeError(
                f"step '{step.name}' failed with exit code {returncode} "
                f"(see {log_path})"
            )


# ---------------------------------------------------------------------------
# SLURM execution
# ---------------------------------------------------------------------------
@dataclass
class SlurmResources:
    partition: str
    cpus: int
    mem: str
    time: str
    gres: str | None = None
    account: str | None = None
    array: str | None = None
    extra_directives: tuple[str, ...] = ()


class SlurmExecutor:
    def __init__(self, slurm_dir: str, logs_dir: str, dry_run: bool = False) -> None:
        self.slurm_dir = slurm_dir
        self.logs_dir = logs_dir
        self.dry_run = dry_run

    def _script_path(self, step: Step) -> str:
        return os.path.join(self.slurm_dir, f"{step.name}.sbatch")

    def render_sbatch(self, step: Step, res: SlurmResources) -> str:
        is_array = res.array is not None
        suffix = "%A_%a" if is_array else "%j"
        out_pattern = os.path.join(self.logs_dir, f"{step.name}-{suffix}.out")

        directives = [
            f"#SBATCH --job-name={step.name}",
            f"#SBATCH --partition={res.partition}",
            f"#SBATCH --cpus-per-task={res.cpus}",
            f"#SBATCH --mem={res.mem}",
            f"#SBATCH --time={res.time}",
            f"#SBATCH --output={out_pattern}",
        ]
        if res.gres:
            directives.append(f"#SBATCH --gres={res.gres}")
        if res.account:
            directives.append(f"#SBATCH --account={res.account}")
        if res.array:
            directives.append(f"#SBATCH --array={res.array}")
        for extra in res.extra_directives:
            directives.append(f"#SBATCH {extra}")

        header = "#!/bin/bash\n" + "\n".join(directives) + "\n\n"
        return header + render_body(step)

    def submit(self, step: Step, res: SlurmResources, dependencies: list[str] | None = None) -> str:
        os.makedirs(self.slurm_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        script = self.render_sbatch(step, res)
        script_path = self._script_path(step)
        with open(script_path, "w") as handle:
            handle.write(script)

        dep_args: list[str] = []
        deps_display = [d for d in (dependencies or []) if d]
        real_deps = [d for d in deps_display if not d.startswith("DRYRUN")]
        if real_deps:
            dep_args = [f"--dependency=afterok:{':'.join(real_deps)}"]

        array_note = f" array={res.array}" if res.array else ""
        print(
            f"[slurm] {step.name}  (partition={res.partition}, mem={res.mem}, "
            f"cpus={res.cpus}{(', gres=' + res.gres) if res.gres else ''}{array_note})"
            + (f"  afterok:{deps_display}" if deps_display else ""),
            flush=True,
        )
        print(f"        script: {script_path}", flush=True)

        if self.dry_run:
            return f"DRYRUN_{step.name}"

        cmd = ["sbatch", "--parsable", *dep_args, script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed for '{step.name}': {result.stderr.strip()}"
            )
        # `--parsable` prints "<jobid>" or "<jobid>;<cluster>".
        job_id = result.stdout.strip().split(";")[0]
        print(f"        submitted: job {job_id}", flush=True)
        return job_id


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
