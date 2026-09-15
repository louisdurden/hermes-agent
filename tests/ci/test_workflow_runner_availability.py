from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_required_workflows_use_runners_available_to_the_fork() -> None:
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )

    assert "ubuntu-latest-96-core" not in workflows
    assert "ubuntu-latest-32-core" not in workflows
    assert "windows-latest-32-core" not in workflows


def test_ci_comment_waits_for_completion_without_holding_a_runner() -> None:
    workflow = _workflow("ci-review-comment.yml")

    assert "types: [completed]" in workflow
    assert "types: [in_progress]" not in workflow