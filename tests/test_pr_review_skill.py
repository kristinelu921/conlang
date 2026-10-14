"""Validate the repository's two-pass PR review contract."""

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIRECTORY = REPOSITORY_ROOT / ".agents" / "skills" / "pr-review"


def read_normalized_skill_text() -> str:
    """Return the skill text with whitespace normalized for assertions."""
    skill_text = (SKILL_DIRECTORY / "SKILL.md").read_text()
    return " ".join(skill_text.split())


def test_pr_review_skill_defines_exactly_two_sequential_passes() -> None:
    """Require one correctness pass followed by one cleanup pass."""
    normalized_skill_text = read_normalized_skill_text()

    assert "one correctness pass, then one cleanup pass" in normalized_skill_text
    assert "Pass 1: Correctness" in normalized_skill_text
    assert "Pass 2: Cleanup" in normalized_skill_text
    assert "Never run these passes concurrently" in normalized_skill_text
    assert "exactly two critic passes total" in normalized_skill_text
    assert "Do not run a third critic pass" in normalized_skill_text
    assert "Do not rerun either critic by default" in normalized_skill_text
    assert "four or five independent critic agents" not in normalized_skill_text
    assert "two complete passes from every configured critic role" not in normalized_skill_text


def test_correctness_pass_covers_types_tests_and_runtime_risk() -> None:
    """Fold type, test, OOD, and reliability checks into correctness."""
    normalized_skill_text = read_normalized_skill_text()

    assert "producers, concrete runtime types and shapes" in normalized_skill_text
    assert "security boundaries" in normalized_skill_text
    assert "array shapes and dtypes" in normalized_skill_text
    assert "RNG ownership" in normalized_skill_text
    assert "would fail for the defect they claim to cover" in normalized_skill_text
    assert "oracle independent of the implementation" in normalized_skill_text
    assert "out-of-distribution cases" in normalized_skill_text
    assert "Never expose private code" in normalized_skill_text
    assert "Producer trace for every type or shape claim" in normalized_skill_text


def test_cleanup_pass_enforces_compact_scope_and_clarity() -> None:
    """Keep cleanup compact without weakening useful abstractions."""
    normalized_skill_text = read_normalized_skill_text()

    assert "dead code, duplication, redundant state" in normalized_skill_text
    assert "smallest clear implementation" in normalized_skill_text
    assert "do not compress readable code into code-golf" in normalized_skill_text
    assert "runtime YAMLs contain only intentional overrides" in normalized_skill_text
    assert "smallest reproducible configuration closure" in normalized_skill_text
    assert "retry, smoke, or abandoned experiment manifests" in normalized_skill_text
    assert "without weakening observable behavior coverage" in normalized_skill_text


def test_pr_review_skill_keeps_critics_read_only_and_coordinator_writing() -> None:
    """Require independent critics and coordinator-owned fixes."""
    normalized_skill_text = read_normalized_skill_text()

    assert "Make the coordinating agent the sole writer" in normalized_skill_text
    assert "one read-only correctness critic" in normalized_skill_text
    assert "one independent read-only cleanup critic" in normalized_skill_text
    assert "Do not reveal the correctness ledger" in normalized_skill_text
    assert "do not allow overlapping agent writes" in normalized_skill_text
    assert "Run narrow tests after behaviorally meaningful fixes" in normalized_skill_text
    assert "final coordinator-owned diff and scope check" in normalized_skill_text
    assert "not a third critic pass" in normalized_skill_text


def test_pr_review_skill_requires_evidence_and_two_row_readiness() -> None:
    """Gate readiness on the two current passes and verified findings."""
    normalized_skill_text = read_normalized_skill_text()

    assert "Verify every Critical and Major finding" in normalized_skill_text
    assert "Producer trace for every type or shape claim" in normalized_skill_text
    assert "exactly two rows: correctness and cleanup" in normalized_skill_text
    assert "Mark an authorized PR READY only when both passes are current" in normalized_skill_text
    assert "report READY_RECOMMENDED" in normalized_skill_text
    assert "leave the PR draft and report NOT_READY" in normalized_skill_text
    assert "no confirmed Critical or Major finding remains" in normalized_skill_text


def test_pr_review_skill_skips_passes_only_for_verified_no_code_diffs() -> None:
    """Allow skips only after semantic no-code classification."""
    normalized_skill_text = read_normalized_skill_text()

    assert "do not classify by filename or extension alone" in normalized_skill_text
    assert "runtime configuration, feature flags, schemas, migrations" in normalized_skill_text
    assert "dependency manifests, lockfiles" in normalized_skill_text
    assert "When uncertain, treat it as code-bearing" in normalized_skill_text
    assert "skip both critic passes" in normalized_skill_text
    assert "SKIPPED_NO_CODE_ITEMS" in normalized_skill_text
    assert "run relevant lightweight checks" in normalized_skill_text


def test_pr_review_skill_metadata_matches_two_pass_workflow() -> None:
    """Keep the skill's user-facing metadata current."""
    metadata_path = SKILL_DIRECTORY / "agents" / "openai.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())

    assert metadata == {
        "interface": {
            "display_name": "PR Review",
            "short_description": "Run correctness then cleanup PR reviews",
            "default_prompt": (
                "Use $pr-review to review this pull request with one "
                "correctness pass followed by one cleanup pass."
            ),
        }
    }
