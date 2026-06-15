"""Prepare or submit Laguna Slurm jobs for transcript CSV generation.

This script is intentionally safe for notebooks: by default it only writes and
prints a Slurm script. It consumes cluster compute only when ``--submit`` is
provided.

Example:
    python scripts/laguna_submit.py \
        --session session_01 \
        --time 04:00:00 \
        --gpus 1 \
        -- --config configs/project_template.yaml

Submit only after reviewing the printed script:
    python scripts/laguna_submit.py --submit --session session_01 -- \
        --config configs/project_template.yaml
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SlurmOptions:
    session: str
    job_name: str
    partition: str
    account: str
    qos: str
    time: str
    cpus: int
    mem: str
    gpus: int
    workdir: str
    venv: str
    conda_env: str
    modules: tuple[str, ...]
    job_dir: Path


def build_slurm_script(options: SlurmOptions, transcribe_args: list[str]) -> str:
    """Build the Slurm script text without submitting it."""
    if not transcribe_args:
        raise ValueError("Provide transcribe_to_csv arguments after '--'.")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={options.job_name}",
        f"#SBATCH --output={options.job_dir}/%x_%j.out",
        f"#SBATCH --error={options.job_dir}/%x_%j.err",
        f"#SBATCH --time={options.time}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={options.cpus}",
        f"#SBATCH --mem={options.mem}",
        f"#SBATCH --partition={options.partition}",
    ]
    if options.account:
        lines.append(f"#SBATCH --account={options.account}")
    if options.qos:
        lines.append(f"#SBATCH --qos={options.qos}")
    if options.gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{options.gpus}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f"WORKDIR={shlex.quote(options.workdir)}",
            f"SESSION={shlex.quote(options.session)}",
            'echo "Laguna transcript job ${SLURM_JOB_ID:-manual} for ${SESSION}"',
            'echo "Working directory: ${WORKDIR}"',
            'cd "${WORKDIR}"',
            f"mkdir -p {shlex.quote(str(options.job_dir))} output/{shlex.quote(options.session)}",
        ]
    )

    for module in options.modules:
        lines.append(f"module load {shlex.quote(module)}")

    if options.venv:
        lines.append(f"source {shlex.quote(options.venv)}/bin/activate")
    elif options.conda_env:
        lines.extend(
            [
                'source "$(conda info --base)/etc/profile.d/conda.sh"',
                f"conda activate {shlex.quote(options.conda_env)}",
            ]
        )

    command = ["python3", "scripts/transcribe_to_csv.py", *transcribe_args]
    lines.extend(
        [
            'export PYTHONPATH="${WORKDIR}:${WORKDIR}/src:${PYTHONPATH:-}"',
            f"echo Running: {shell_join(command)}",
            shell_join(command),
            'echo "Done: output/${SESSION}/utterances.csv (or the --out path you set)"',
            "",
        ]
    )
    return "\n".join(lines)


def shell_join(parts: list[str]) -> str:
    """Quote a shell command for display and execution in the Slurm script."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch")
    parser.add_argument("--session", default="session_01")
    parser.add_argument("--job-name", default="", help="Defaults to transcript-<session>")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--account", default="")
    parser.add_argument("--qos", default="")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workdir", default=os.environ.get("TRANSCRIPT_REPO", str(Path.cwd())))
    parser.add_argument("--venv", default=os.environ.get("TRANSCRIPT_VENV", ""))
    parser.add_argument("--conda-env", default=os.environ.get("TRANSCRIPT_CONDA_ENV", ""))
    parser.add_argument("--module", action="append", default=[], dest="modules")
    parser.add_argument("--job-dir", default="output/slurm")
    parser.add_argument("transcribe_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    transcribe_args = _strip_remainder_marker(args.transcribe_args)
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    options = SlurmOptions(
        session=args.session,
        job_name=args.job_name or f"transcript-{args.session}",
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        time=args.time,
        cpus=args.cpus,
        mem=args.mem,
        gpus=args.gpus,
        workdir=args.workdir,
        venv=args.venv,
        conda_env=args.conda_env,
        modules=tuple(args.modules),
        job_dir=job_dir,
    )
    script = build_slurm_script(options, transcribe_args)
    script_path = _script_path(job_dir, options.session)
    script_path.write_text(script, encoding="utf-8")

    print(f"[laguna_submit] wrote {script_path}")
    print(script)

    if not args.submit:
        print("[laguna_submit] dry run only. Add --submit to call sbatch.")
        return 0

    result = subprocess.run(
        ["sbatch", str(script_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    print(result.stdout.strip())
    return 0


def _strip_remainder_marker(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _script_path(job_dir: Path, session: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session)
    return job_dir / f"transcribe_{safe_session}_{stamp}.slurm"


if __name__ == "__main__":
    raise SystemExit(main())
