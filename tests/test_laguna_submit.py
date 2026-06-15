from pathlib import Path

from scripts.laguna_submit import SlurmOptions, build_slurm_script, main


def test_build_slurm_script_targets_transcript_tool(tmp_path):
    options = SlurmOptions(
        session="session_01",
        job_name="transcript-session_01",
        partition="gpu",
        account="",
        qos="",
        time="04:00:00",
        cpus=8,
        mem="32G",
        gpus=1,
        workdir="/cluster/home/user/behavior-pipeline",
        venv="/cluster/home/user/behavior-pipeline/transcript-env",
        conda_env="",
        modules=("cuda/12.1",),
        job_dir=tmp_path,
    )

    script = build_slurm_script(
        options,
        ["--config", "configs/project_template.yaml"],
    )

    assert "#SBATCH --gres=gpu:1" in script
    assert "module load cuda/12.1" in script
    assert "source /cluster/home/user/behavior-pipeline/transcript-env/bin/activate" in script
    assert "python3 scripts/transcribe_to_csv.py --config configs/project_template.yaml" in script


def test_laguna_submit_dry_run_writes_script_without_sbatch(tmp_path, capsys):
    exit_code = main(
        [
            "--session",
            "session_01",
            "--job-dir",
            str(tmp_path),
            "--workdir",
            "/repo",
            "--gpus",
            "0",
            "--",
            "--from-json",
            "output/session_01/video_turns.json",
            "--out",
            "output/session_01/utterances.csv",
        ]
    )

    assert exit_code == 0
    scripts = list(Path(tmp_path).glob("transcribe_session_01_*.slurm"))
    assert len(scripts) == 1
    text = scripts[0].read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu" not in text
    assert "--from-json output/session_01/video_turns.json" in text
    assert "dry run only" in capsys.readouterr().out
