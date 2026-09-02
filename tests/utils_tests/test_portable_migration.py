"""Tests for the sealed, configuration-driven workstation migration."""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATION_DIRECTORY = ROOT / "scripts" / "migration"
sys.path.insert(0, str(MIGRATION_DIRECTORY))

portable = importlib.import_module("portable_workspace")
secret_scan = importlib.import_module("scan_portable_secrets")


CONFIG_PATH = MIGRATION_DIRECTORY / "portable_config.json"


def run_git(repository: Path, *arguments: str) -> str:
    """Run one test Git command and return stripped UTF-8 stdout."""

    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout.decode("utf-8").strip()


def synthetic_assigned_secret() -> str:
    """Build a scanner fixture without embedding its complete pattern in Git."""

    return "pass" + "word=" + "abcdefgh" + "ijklmnop"


def initialize_repository(repository: Path) -> str:
    """Create a one-commit main branch with deterministic local identity."""

    repository.mkdir()
    run_git(repository, "init")
    run_git(repository, "config", "user.name", "Portable Test")
    run_git(repository, "config", "user.email", "portable@example.invalid")
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    run_git(repository, "add", "tracked.txt")
    run_git(repository, "commit", "-m", "base")
    run_git(repository, "branch", "-M", "main")
    return run_git(repository, "rev-parse", "HEAD")


def read_config_value() -> dict[str, object]:
    """Read a fresh mutable copy of the tracked migration config."""

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_config_value(path: Path, value: dict[str, object]) -> None:
    """Write one temporary JSON config for a validation test."""

    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def minimal_repository_inventory() -> tuple[object, dict[str, object]]:
    """Build the smallest internally consistent repository inventory."""

    config = portable.load_config_file(CONFIG_PATH)
    final_tip = "a" * 40
    final_tree = "b" * 40
    source_tree = "c" * 40
    bundle_hash = "d" * 64
    empty_hash = hashlib.sha256(b"").hexdigest()
    migration_ref = f"refs/heads/{config.repository['migration_branch']}"
    refs = sorted(
        [
            {"refname": migration_ref, "oid": final_tip, "symref": ""},
            {
                "refname": config.repository["local_main_ref"],
                "oid": config.repository["pre_migration_main"],
                "symref": "",
            },
        ],
        key=lambda item: item["refname"],
    )
    worktree = {
        "path": "C:/portable/repository",
        "head": final_tip,
        "branch": migration_ref,
        "detached": False,
        "bare": False,
        "locked": False,
        "prunable": False,
        "status": [],
        "status_display": [],
        "status_porcelain_v1_z_sha256": empty_hash,
        "head_patch_sha256": empty_hash,
        "index_patch_sha256": empty_hash,
        "containing_refs": [migration_ref],
        "snapshot_path": None,
        "snapshot_files": [],
        "ignored_inventory": {
            "ignored_path_count": 0,
            "ignored_paths_sha256": empty_hash,
            "selected_context_ignored_path_count": 0,
            "selected_context_ignored_paths_sha256": empty_hash,
            "excluded_ignored_path_count": 0,
            "excluded_ignored_paths_sha256": empty_hash,
        },
    }
    policy = {
        "continuing_branch": config.repository["continuing_branch"],
        "local_main_ref": config.repository["local_main_ref"],
        "pre_migration_main_archive_ref": config.repository[
            "pre_migration_main_archive_ref"
        ],
        "remote_name": config.repository["remote_name"],
        "restore_bare_repository_directory": config.repository[
            "restore_bare_repository_directory"
        ],
        "restore_checkout_directory": config.repository[
            "restore_checkout_directory"
        ],
    }
    inventory = {
        "schema_version": 1,
        "created_at_utc": "2026-08-14T00:00:00Z",
        "config_sha256": config.sha256,
        "config_repository_path": config.archive["config_repository_path"],
        "source_base": config.repository["reviewed_source_base"],
        "source_base_tree": source_tree,
        "final_tip": final_tip,
        "final_tip_tree": final_tree,
        "pre_migration_main": config.repository["pre_migration_main"],
        "observed_main": config.repository["pre_migration_main"],
        "export_head_symbolic_ref": migration_ref,
        "restored_head_symbolic_ref": config.repository["local_main_ref"],
        "repository_policy": policy,
        "changed_paths_after_source": [],
        "remote_url": None,
        "refs": refs,
        "archived_worktree_refs": [],
        "worktrees": [worktree],
        "bundle_heads": [
            {"refname": "HEAD", "oid": final_tip},
            *[
                {"refname": ref["refname"], "oid": ref["oid"]}
                for ref in refs
            ],
        ],
        "bundle_sha256": bundle_hash,
        "bundle_verification": {
            "bundle_verify_output_sha256": empty_hash,
            "fresh_mirror_ref_count": len(refs),
            "fresh_mirror_exact_ref_match": True,
            "reconstructed_symbolic_refs": [],
        },
    }
    return config, inventory


def test_project_policy_is_loaded_only_from_config() -> None:
    """Project identities must not be duplicated in the generic engine."""

    config = portable.load_config_file(CONFIG_PATH)
    source = (MIGRATION_DIRECTORY / "portable_workspace.py").read_text(
        encoding="utf-8"
    )
    project_values = [
        config.repository["reviewed_source_base"],
        config.repository["pre_migration_main"],
        config.repository["migration_branch"],
        config.handoff["historical_branch"],
        *(item["id"] for item in config.context["codex_sessions"]),
        *(item["id"] for item in config.context["claude_sessions"]),
    ]

    assert all(value not in source for value in project_values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unknown": 1}), "keys do not match"),
        (lambda value: value.update({"schema_version": True}), "unsupported"),
        (
            lambda value: value["archive_policy"]["excluded_components"].remove(
                "datasets"
            ),
            "weaken component safety",
        ),
        (
            lambda value: value["archive_policy"].update(
                {"max_archive_expanded_bytes": 4 * 1024 * 1024 * 1024 + 1}
            ),
            "exceeds the safety maximum",
        ),
        (
            lambda value: value["context_policy"]["codex_memory_files"].reverse(),
            "sorted and unique",
        ),
        (
            lambda value: value["context_policy"]["codex_sessions"][0].update(
                {"path": "../session.jsonl"}
            ),
            "non-normalized archive path",
        ),
        (
            lambda value: value["handoff_policy"].update(
                {"reviewed_result_sha256": "not-a-sha"}
            ),
            "is not a SHA-256",
        ),
        (
            lambda value: value["context_policy"].update(
                {
                    "codex_sessions": [
                        {"id": "A", "path": "sessions/A.jsonl"},
                        {"id": "a", "path": "sessions/a.jsonl"},
                    ]
                }
            ),
            "case-insensitively unique",
        ),
        (
            lambda value: value["repository_policy"].update(
                {
                    "restore_bare_repository_directory": "same",
                    "restore_checkout_directory": "SAME",
                }
            ),
            "directories collide",
        ),
        (
            lambda value: value["repository_policy"].update(
                {"restore_bare_repository_directory": "archive-content"}
            ),
            "protocol names collide",
        ),
        (
            lambda value: value["archive_policy"].update(
                {"archive_name": "RESTORE.txt"}
            ),
            "protocol filenames collide",
        ),
        (
            lambda value: value["archive_policy"].update(
                {"bundle_relative_path": "MANIFEST.json"}
            ),
            "protocol paths collide",
        ),
    ],
)
def test_config_rejects_schema_and_policy_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    """Unknown, unsafe, or noncanonical config edits must fail closed."""

    value = copy.deepcopy(read_config_value())
    mutation(value)
    path = tmp_path / "portable_config.json"
    write_config_value(path, value)

    with pytest.raises(portable.PortableWorkspaceError, match=message):
        portable.load_config_file(path)


@pytest.mark.parametrize(
    ("selector", "existing_key"),
    [
        (lambda value: value, "schema_version"),
        (lambda value: value["repository_policy"], "remote_name"),
        (lambda value: value["archive_policy"], "archive_name"),
        (lambda value: value["context_policy"], "agent_context_root"),
        (lambda value: value["handoff_policy"], "review_task"),
        (
            lambda value: value["archive_policy"]["restore_tools"],
            "scanner",
        ),
        (
            lambda value: value["context_policy"]["settings_templates"]["codex"],
            "source",
        ),
        (
            lambda value: value["context_policy"]["codex_sessions"][0],
            "path",
        ),
    ],
)
@pytest.mark.parametrize("operation", ["missing", "unknown"])
def test_config_requires_exact_keys_at_every_schema_layer(
    tmp_path: Path,
    selector,
    existing_key: str,
    operation: str,
) -> None:
    """Every nested config mapping rejects missing and unknown fields."""

    value = copy.deepcopy(read_config_value())
    mapping = selector(value)
    if operation == "missing":
        mapping.pop(existing_key)
    else:
        mapping["unexpected_key"] = "unexpected"
    path = tmp_path / "portable_config.json"
    write_config_value(path, value)

    with pytest.raises(portable.PortableWorkspaceError, match="keys do not match"):
        portable.load_config_file(path)


def test_config_parse_and_hash_use_one_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config parsing and identity must come from the same file read."""

    path = tmp_path / "portable_config.json"
    snapshot = CONFIG_PATH.read_bytes()
    path.write_bytes(snapshot)
    original_read_bytes = Path.read_bytes
    reads = 0

    def read_bytes_once(self: Path) -> bytes:
        nonlocal reads
        if self == path:
            reads += 1
            if reads > 1:
                return b"{}\n"
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)

    config = portable.load_config_file(path)

    assert reads == 1
    assert config.sha256 == hashlib.sha256(snapshot).hexdigest()


def test_config_unchanged_rejects_byte_only_drift(tmp_path: Path) -> None:
    """Whitespace-only config rewrites still invalidate the sealed operation."""

    path = tmp_path / "portable_config.json"
    path.write_bytes(CONFIG_PATH.read_bytes())
    config = portable.load_config_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(portable.PortableWorkspaceError, match="config changed"):
        portable.require_config_unchanged(config)


def test_config_loader_rejects_symlink_input(tmp_path: Path) -> None:
    """Config identity cannot be redirected before its bytes are read."""

    source = tmp_path / "source.json"
    source.write_bytes(CONFIG_PATH.read_bytes())
    link = tmp_path / "portable_config.json"
    try:
        link.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(portable.PortableWorkspaceError, match="links are not portable"):
        portable.load_config_file(link)


def test_external_json_accepts_explicit_utf8_bom() -> None:
    """PowerShell-authored JSON evidence may carry one UTF-8 byte-order mark."""

    assert portable.parse_external_json(b"\xef\xbb\xbf{\"status\":\"ok\"}\n", "job") == {
        "status": "ok"
    }


def handoff_fixture() -> tuple[bytes, bytes, dict[str, object], str]:
    """Build one internally bound successful handoff evidence pair."""

    config = portable.load_config_file(CONFIG_PATH)
    policy = copy.deepcopy(config.handoff)
    source_base = "1" * 40
    metrics = {
        "completed_epochs": 20,
        "train_updates": 780,
        "direction_refreshes": 780,
        "direction_example_visits": 3_120_000,
        "scored_rows": 99_840,
        "applied_rows": 0,
        "action_batches": 0,
        "vdev_batches": 640,
        "vtest_batches": 0,
        "checkpoint_files": 0,
        "stable_epoch_count": 19,
        "epoch_1_loop_wall_seconds": 97.3,
        "stable_epoch_wall_mean_seconds": 5.2,
        "stable_epoch_wall_median_seconds": 5.2,
        "stable_epoch_wall_p90_seconds": 5.4,
        "wrapper_wall_seconds": 245.4,
        "completion_file_sha256": policy["completion_artifact_sha256"],
    }
    job = {
        "commit": source_base,
        "dataset": policy["expected_job_values"]["dataset"],
        "method": policy["expected_job_values"]["method"],
        "policy_mode": policy["expected_job_values"]["policy_mode"],
        "split_protocol": "sealed split",
        "status": policy["required_job_status"],
        "tracking_url": "https://example.invalid/run/1",
        "metrics": metrics,
    }
    job_snapshot = json.dumps(job).encode()
    job_sha256 = hashlib.sha256(job_snapshot).hexdigest()
    policy["reviewed_remote_job_sha256"] = job_sha256
    review_text = "\n".join(
        [
            f"Task: {policy['review_task']}",
            "Type: REVIEW",
            "Status: RESOLVED",
            job_sha256,
            policy["completion_artifact_path"],
            policy["completion_artifact_sha256"],
        ]
    )
    review_snapshot = review_text.encode()
    policy["reviewed_result_sha256"] = hashlib.sha256(review_snapshot).hexdigest()
    return job_snapshot, review_snapshot, policy, source_base


def test_handoff_evidence_accepts_exact_bound_success() -> None:
    """The sealed job and resolved review must validate as one evidence pair."""

    job, review, policy, source_base = handoff_fixture()

    observed_job, metrics, review_text = portable.validate_handoff_evidence(
        job,
        review,
        policy,
        source_base,
    )

    assert observed_job["status"] == "SUCCESS"
    assert metrics["completed_epochs"] == 20
    assert "Status: RESOLVED" in review_text


@pytest.mark.parametrize(
    "mutation",
    [
        "job_hash",
        "review_hash",
        "review_metadata",
        "review_completion",
        "nan_timing",
    ],
)
def test_handoff_evidence_rejects_broken_binding(mutation: str) -> None:
    """Any byte, review, completion, or finite-metric mismatch must fail closed."""

    job, review, policy, source_base = handoff_fixture()
    if mutation == "job_hash":
        job += b"\n"
    elif mutation == "review_hash":
        review += b"\n"
    elif mutation == "review_metadata":
        review = review.replace(b"Type: REVIEW", b"Type: INFO")
        policy["reviewed_result_sha256"] = hashlib.sha256(review).hexdigest()
    elif mutation == "review_completion":
        review = review.replace(policy["completion_artifact_path"].encode(), b"missing")
        policy["reviewed_result_sha256"] = hashlib.sha256(review).hexdigest()
    else:
        value = json.loads(job)
        value["metrics"]["wrapper_wall_seconds"] = float("nan")
        job = json.dumps(value).encode()
        policy["reviewed_remote_job_sha256"] = hashlib.sha256(job).hexdigest()

    with pytest.raises(portable.PortableWorkspaceError):
        portable.validate_handoff_evidence(job, review, policy, source_base)


def test_config_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """Duplicate keys must not override an earlier configuration value."""

    snapshot = CONFIG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "portable_config.json"
    path.write_text(
        snapshot.replace(
            '"schema_version": 1,',
            '"schema_version": 2,\n  "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(portable.PortableWorkspaceError, match="duplicate JSON key"):
        portable.load_config_file(path)


@pytest.mark.parametrize("kind", ["codex", "claude"])
def test_transcript_export_rejects_zero_selected_messages(
    tmp_path: Path,
    kind: str,
) -> None:
    """A selected session must contribute at least one user or assistant message."""

    source = tmp_path / "empty.jsonl"
    source.write_text('{"type":"tool"}\n', encoding="utf-8")
    destination = tmp_path / "transcript.md"

    with pytest.raises(portable.PortableWorkspaceError, match="no selected messages"):
        if kind == "codex":
            portable.extract_codex_transcript(source, "thread", destination)
        else:
            portable.extract_claude_transcript(source, "thread", destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "value",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "folder\\file",
        "folder//file",
        "folder/./file",
        "name:stream",
        "NUL.txt",
        "COM¹.txt",
        "LPT³.log",
        "trailing.",
        "trailing ",
        "control\x01name",
        "bad?.txt",
        "bad*.txt",
        "bad<.txt",
        "bad>.txt",
        'bad".txt',
        "bad|.txt",
    ],
)
def test_safe_relative_path_rejects_nonportable_names(value: str) -> None:
    """Archive paths must be safe on a new Windows workstation."""

    with pytest.raises(portable.PortableWorkspaceError):
        portable.safe_relative_path(value)


def test_safe_relative_path_enforces_windows_component_units() -> None:
    """Each component must fit the Windows 255 UTF-16-unit filename limit."""

    astral = "\U0001f642"
    assert portable.safe_relative_path("a" * 255).as_posix() == "a" * 255
    assert portable.safe_relative_path(astral * 127).as_posix() == astral * 127
    for value in ("a" * 256, astral * 128):
        with pytest.raises(portable.PortableWorkspaceError, match="255 UTF-16"):
            portable.safe_relative_path(value)


def test_secret_scanner_rejects_env_tokens_and_utf16(tmp_path: Path) -> None:
    """Credential filenames and high-confidence token text must be detected."""

    (tmp_path / ".env.local").write_text("placeholder\n", encoding="utf-8")
    token = "ghp_" + "A" * 24
    (tmp_path / "token.txt").write_text(token + "\n", encoding="utf-8")
    utf16_secret = "api_key='" + "B" * 24 + "'"
    (tmp_path / "utf16.txt").write_bytes(utf16_secret.encode("utf-16"))

    findings = secret_scan.scan_tree(tmp_path)

    assert any(".env.local:credential_filename" in item for item in findings)
    assert any("token.txt:github_token" in item for item in findings)
    assert any("utf16.txt:assigned_secret" in item for item in findings)


def test_secret_scanner_test_source_contains_no_complete_fixture() -> None:
    """Scanner fixtures must be assembled at runtime, not stored in Git blobs."""

    assert secret_scan._secret_kinds(Path(__file__).read_bytes()) == []


@pytest.mark.parametrize(
    "payload",
    [
        synthetic_assigned_secret().encode("ascii"),
        ("api_key='" + "A" * 24 + "'").encode("utf-16-le"),
        ("ghp_" + "B" * 24).encode("utf-16-le"),
    ],
)
def test_secret_scanner_rejects_unquoted_and_bomless_utf16(payload: bytes) -> None:
    """Unquoted assignments and BOM-less UTF-16 tokens must be detected."""

    assert secret_scan._secret_kinds(payload)


def test_secret_scanner_checks_historical_git_paths(tmp_path: Path) -> None:
    """A credential filename in reachable history must block the bundle."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    (repository / ".env").write_text("placeholder\n", encoding="utf-8")
    run_git(repository, "add", ".env")
    run_git(repository, "commit", "-m", "credential path")
    (repository / ".env").unlink()
    run_git(repository, "add", "-u")
    run_git(repository, "commit", "-m", "remove credential path")

    findings = secret_scan.scan_git(repository)

    assert any(":.env:credential_filename" in item for item in findings)


def test_secret_scanner_redacts_tokens_embedded_in_paths(tmp_path: Path) -> None:
    """Credential-shaped path text must be rejected without echoing it."""

    token_name = "ghp_" + "P" * 24 + ".txt"
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / token_name).write_text("benign\n", encoding="utf-8")
    tree_findings = secret_scan.scan_tree(tree)
    assert any("path_github_token" in item for item in tree_findings)
    assert token_name not in "\n".join(tree_findings)

    repository = tmp_path / "repository"
    initialize_repository(repository)
    (repository / token_name).write_text("benign\n", encoding="utf-8")
    run_git(repository, "add", token_name)
    run_git(repository, "commit", "-m", "path scan")
    git_findings = secret_scan.scan_git(repository)
    assert any("path_github_token" in item for item in git_findings)
    assert token_name not in "\n".join(git_findings)


def test_secret_scanner_ignores_replace_object_substitution(tmp_path: Path) -> None:
    """Replacement refs cannot hide original reachable credential bytes."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    (repository / "secret.txt").write_text(
        synthetic_assigned_secret() + "\n", encoding="utf-8"
    )
    run_git(repository, "add", "secret.txt")
    run_git(repository, "commit", "-m", "original object")
    secret_oid = run_git(repository, "rev-parse", "HEAD:secret.txt")
    benign_oid = portable.decode_utf8(
        portable.run_git(
            repository,
            "hash-object",
            "-w",
            "--stdin",
            input_bytes=b"benign\n",
        ).stdout.strip(),
        "test benign object",
    )
    run_git(repository, "replace", secret_oid, benign_oid)

    assert any("assigned_secret" in item for item in secret_scan.scan_git(repository))
    with pytest.raises(portable.PortableWorkspaceError, match="replacement refs"):
        portable.assert_repository_self_contained(
            repository, run_git(repository, "rev-parse", "HEAD")
        )


def test_secret_scanner_rejects_git_lfs_pointers(tmp_path: Path) -> None:
    """External LFS payload references cannot be represented by a Git bundle alone."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    (repository / "large.bin").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 123\n",
        encoding="utf-8",
    )
    run_git(repository, "add", "large.bin")
    run_git(repository, "commit", "-m", "add external pointer")

    assert any("git_lfs_pointer" in item for item in secret_scan.scan_git(repository))


@pytest.mark.parametrize(
    ("object_kind", "secret"),
    [
        ("commit", "ghp_" + "C" * 24),
        ("tag", synthetic_assigned_secret()),
    ],
)
def test_secret_scanner_checks_commit_and_tag_messages(
    tmp_path: Path,
    object_kind: str,
    secret: str,
) -> None:
    """Reachable commit and annotated-tag messages must pass the secret gate."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    if object_kind == "commit":
        run_git(repository, "commit", "--allow-empty", "-m", secret)
    else:
        run_git(repository, "tag", "-a", "secret-tag", "-m", secret)

    findings = secret_scan.scan_git(repository)

    assert findings


@pytest.mark.parametrize("suffix", [".zip", ".bundle"])
def test_secret_scanner_rejects_unapproved_opaque_archives(
    tmp_path: Path,
    suffix: str,
) -> None:
    """A dirty snapshot cannot hide credential bytes in an opaque container."""

    path = tmp_path / f"untracked{suffix}"
    path.write_bytes(synthetic_assigned_secret().encode("ascii"))

    assert secret_scan.scan_tree(tmp_path) == [
        f"untracked{suffix}:opaque_archive"
    ]


@pytest.mark.parametrize("name", ["session.jsonl.gz", "array.npy.gz"])
def test_archive_rejects_nested_raw_suffixes_and_compressed_content(
    tmp_path: Path,
    name: str,
) -> None:
    """Compression cannot hide a raw session or array suffix from exclusion gates."""

    policy = portable.load_config_file(CONFIG_PATH).archive
    with pytest.raises(portable.PortableWorkspaceError, match="excluded raw file type"):
        portable.assert_archive_exclusions([f"snapshot/{name}"], policy)
    payload = tmp_path / "payload.bin"
    payload.write_bytes(gzip.compress(synthetic_assigned_secret().encode("ascii")))
    assert secret_scan.scan_tree(tmp_path) == ["payload.bin:opaque_archive"]


def test_secret_scanner_rejects_renamed_zip_payload(tmp_path: Path) -> None:
    """ZIP magic remains opaque when the filename has an unrelated suffix."""

    payload = tmp_path / "payload.bin"
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("secret.txt", synthetic_assigned_secret())

    assert secret_scan.scan_tree(tmp_path) == ["payload.bin:opaque_archive"]


def write_raw_zip(
    path: Path,
    entries: list[tuple[zipfile.ZipInfo | str, bytes]],
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    """Write a small raw ZIP for hostile-layout tests."""

    with zipfile.ZipFile(path, "w", compression=compression) as handle:
        for name, data in entries:
            if isinstance(name, zipfile.ZipInfo):
                name.compress_type = compression
            handle.writestr(name, data)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "folder\\file",
        "name:stream",
        "AUX.txt",
        "trailing.",
    ],
)
def test_zip_inspection_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    """Unsafe ZIP member names must fail before extraction."""

    archive = tmp_path / "bad.zip"
    write_raw_zip(archive, [(name, b"payload")])
    policy = portable.load_config_file(CONFIG_PATH).archive

    with pytest.raises(portable.PortableWorkspaceError):
        portable.inspect_zip_archive(archive, policy)


def test_zip_inspection_rejects_case_collisions_and_file_parents(
    tmp_path: Path,
) -> None:
    """ZIP entries cannot collide by case or serve as another file's parent."""

    policy = portable.load_config_file(CONFIG_PATH).archive
    case_archive = tmp_path / "case.zip"
    write_raw_zip(case_archive, [("same", b"a"), ("SAME", b"b")])
    with pytest.raises(portable.PortableWorkspaceError, match="duplicate ZIP path"):
        portable.inspect_zip_archive(case_archive, policy)

    parent_archive = tmp_path / "parent.zip"
    write_raw_zip(parent_archive, [("Node", b"a"), ("node/child", b"b")])
    with pytest.raises(portable.PortableWorkspaceError, match="also a parent"):
        portable.inspect_zip_archive(parent_archive, policy)


def test_zip_inspection_rejects_links_and_expanded_size(tmp_path: Path) -> None:
    """Linked members and archives over the configured bound must fail."""

    policy = copy.deepcopy(portable.load_config_file(CONFIG_PATH).archive)
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    link_archive = tmp_path / "link.zip"
    write_raw_zip(link_archive, [(link, b"target")])
    with pytest.raises(portable.PortableWorkspaceError, match="linked ZIP"):
        portable.inspect_zip_archive(link_archive, policy)

    fifo = zipfile.ZipInfo("fifo")
    fifo.create_system = 3
    fifo.external_attr = (stat.S_IFIFO | 0o644) << 16
    fifo_archive = tmp_path / "fifo.zip"
    write_raw_zip(fifo_archive, [(fifo, b"payload")])
    with pytest.raises(portable.PortableWorkspaceError, match="unsupported ZIP entry"):
        portable.inspect_zip_archive(fifo_archive, policy)

    policy["max_archive_expanded_bytes"] = 2
    large_archive = tmp_path / "large.zip"
    write_raw_zip(large_archive, [("payload", b"abc")])
    with pytest.raises(portable.PortableWorkspaceError, match="expanded-size"):
        portable.inspect_zip_archive(large_archive, policy)


@pytest.mark.parametrize(
    "compression",
    [zipfile.ZIP_STORED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA],
)
def test_zip_inspection_rejects_nonprotocol_compression(
    tmp_path: Path,
    compression: int,
) -> None:
    """Restore accepts only the DEFLATE method emitted by the exporter."""

    archive = tmp_path / "compression.zip"
    write_raw_zip(archive, [("payload", b"content")], compression=compression)
    policy = portable.load_config_file(CONFIG_PATH).archive

    with pytest.raises(portable.PortableWorkspaceError, match="compression method"):
        portable.inspect_zip_archive(archive, policy)


def test_manifest_and_zip_round_trip(tmp_path: Path) -> None:
    """A generated archive must pass its own size and digest verification."""

    config = portable.load_config_file(CONFIG_PATH)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "payload.txt").write_text("portable\n", encoding="utf-8")
    final_tip = "1" * 40
    manifest_sha, count = portable.write_manifest(
        staging,
        final_tip,
        config.sha256,
        config.archive,
    )
    archive = tmp_path / config.archive["archive_name"]
    portable.create_zip(
        staging,
        archive,
        config.archive["zip_compression_level"],
    )

    extracted = tmp_path / "extracted"
    manifest, observed_sha = portable.inspect_zip_archive(
        archive,
        config.archive,
        extract_destination=extracted,
    )

    assert count == 1
    assert observed_sha == manifest_sha
    assert manifest["final_tip"] == final_tip
    assert manifest["config_sha256"] == config.sha256
    assert (extracted / "payload.txt").read_text(encoding="utf-8") == "portable\n"


def test_manifest_writer_enforces_expanded_size_limit(tmp_path: Path) -> None:
    """The exporter must not create a ZIP that its restore path would reject."""

    config = portable.load_config_file(CONFIG_PATH)
    policy = copy.deepcopy(config.archive)
    policy["max_archive_expanded_bytes"] = 2
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "payload").write_bytes(b"abc")

    with pytest.raises(portable.PortableWorkspaceError, match="expanded-size"):
        portable.write_manifest(staging, "1" * 40, config.sha256, policy)


def test_dirty_snapshot_rejects_dangling_symlink(tmp_path: Path) -> None:
    """A missing link target must not make a dirty symlink appear absent."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    link = repository / "dangling"
    try:
        link.symlink_to(repository / "missing-target")
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(portable.PortableWorkspaceError, match="links are not portable"):
        portable.worktree_file_record(repository, "dangling", snapshot, 1024)


def test_tree_walkers_reject_windows_junctions(tmp_path: Path) -> None:
    """Directory junctions must fail before a selected tree can be traversed."""

    if not hasattr(os.path, "isjunction"):
        pytest.skip("junction detection is unavailable")
    source = tmp_path / "source"
    target = tmp_path / "target"
    destination = tmp_path / "destination"
    source.mkdir()
    target.mkdir()
    junction = source / "junction"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        pytest.skip("junction creation is unavailable")
    try:
        assert os.path.isjunction(junction)
        with pytest.raises(portable.PortableWorkspaceError, match="links are not portable"):
            portable.copy_tree_files(source, destination)
        with pytest.raises(portable.PortableWorkspaceError, match="links are not portable"):
            portable.staging_files(source)
        assert secret_scan.scan_tree(source) == ["junction:link"]
    finally:
        junction.rmdir()


def test_ai_context_restore_rejects_linked_destination_root(tmp_path: Path) -> None:
    """Curated context must never be written through a checkout junction."""

    if not hasattr(os.path, "isjunction"):
        pytest.skip("junction detection is unavailable")
    checkout = tmp_path / "checkout"
    initialize_repository(checkout)
    outside = tmp_path / "outside"
    outside.mkdir()
    extracted = tmp_path / "extracted"
    (extracted / "context").mkdir(parents=True)
    config = portable.load_config_file(CONFIG_PATH)
    destination = checkout / config.context["restore_context_directory"]
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(destination), str(outside)],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        pytest.skip("junction creation is unavailable")
    try:
        with pytest.raises(portable.PortableWorkspaceError, match="links are not portable"):
            portable.copy_ai_context_into_checkout(
                extracted, checkout, config.context
            )
        assert list(outside.iterdir()) == []
    finally:
        destination.rmdir()


def test_project_context_restore_rejects_linked_parent(tmp_path: Path) -> None:
    """Project context must never escape through an existing checkout parent."""

    if not hasattr(os.path, "isjunction"):
        pytest.skip("junction detection is unavailable")
    checkout = tmp_path / "checkout"
    initialize_repository(checkout)
    outside = tmp_path / "outside"
    outside.mkdir()
    extracted = tmp_path / "extracted"
    project_file = extracted / "context" / "project" / ".agents" / "notes.md"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("continuity\n", encoding="utf-8")
    destination_parent = checkout / ".agents"
    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(destination_parent),
            str(outside),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        pytest.skip("junction creation is unavailable")
    try:
        with pytest.raises(
            portable.PortableWorkspaceError,
            match="links are not portable|path escapes trusted root",
        ):
            portable.copy_project_context_into_checkout(extracted, checkout)
        assert list(outside.iterdir()) == []
    finally:
        destination_parent.rmdir()


def test_transfer_manifest_rejects_tampering(tmp_path: Path) -> None:
    """Every outer transfer file must remain byte-for-byte unchanged."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one\n", encoding="utf-8")
    second.write_text("two\n", encoding="utf-8")
    digest = portable.write_transfer_manifest(tmp_path, [first, second])

    assert portable.verify_transfer_manifest(
        tmp_path, {"first.txt", "second.txt"}
    ) == digest

    second.write_text("changed\n", encoding="utf-8")
    with pytest.raises(portable.PortableWorkspaceError, match="differs"):
        portable.verify_transfer_manifest(
            tmp_path, {"first.txt", "second.txt"}
        )


def test_transfer_manifest_requires_the_exact_protocol_file_set(
    tmp_path: Path,
) -> None:
    """A hashed extra outer file must still fail the restore protocol."""

    required = tmp_path / "required.txt"
    extra = tmp_path / "extra.txt"
    required.write_text("required\n", encoding="utf-8")
    extra.write_text("extra\n", encoding="utf-8")
    portable.write_transfer_manifest(tmp_path, [required, extra])

    with pytest.raises(portable.PortableWorkspaceError, match="file set differs"):
        portable.verify_transfer_manifest(tmp_path, {"required.txt"})


def test_transfer_manifest_rejects_payload_symlink_before_hashing(
    tmp_path: Path,
) -> None:
    """A manifest entry cannot redirect reads outside the transfer directory."""

    payload = tmp_path / "payload.txt"
    payload.write_text("same bytes\n", encoding="utf-8")
    portable.write_transfer_manifest(tmp_path, [payload])
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("same bytes\n", encoding="utf-8")
    payload.unlink()
    try:
        payload.symlink_to(outside)
    except OSError as error:
        outside.unlink()
        pytest.skip(f"symlink creation is unavailable: {error}")
    try:
        with pytest.raises(
            portable.PortableWorkspaceError, match="links are not portable"
        ):
            portable.verify_transfer_manifest(tmp_path, {"payload.txt"})
    finally:
        payload.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


def test_transfer_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """A rehashed outer manifest cannot override an earlier JSON field."""

    payload = tmp_path / "payload.txt"
    payload.write_text("one\n", encoding="utf-8")
    portable.write_transfer_manifest(tmp_path, [payload])
    manifest = tmp_path / "TRANSFER_MANIFEST.json"
    text = manifest.read_text(encoding="utf-8").replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    manifest.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "TRANSFER_MANIFEST.sha256").write_text(
        f"{digest}  TRANSFER_MANIFEST.json\n",
        encoding="utf-8",
    )

    with pytest.raises(portable.PortableWorkspaceError, match="duplicate JSON key"):
        portable.verify_transfer_manifest(tmp_path, {"payload.txt"})


def test_transfer_manifest_rejects_windows_alternate_streams(tmp_path: Path) -> None:
    """Bytes in an NTFS named stream must not fall outside transfer hashes."""

    if os.name != "nt":
        pytest.skip("NTFS stream inspection is Windows-only")
    payload = tmp_path / "payload.txt"
    payload.write_text("public\n", encoding="utf-8")
    portable.write_transfer_manifest(tmp_path, [payload])
    stream = Path(f"{payload}:hidden")
    try:
        stream.write_text(synthetic_assigned_secret(), encoding="utf-8")
    except OSError as error:
        pytest.skip(f"alternate streams are unavailable: {error}")

    with pytest.raises(portable.PortableWorkspaceError, match="alternate data streams"):
        portable.verify_transfer_manifest(tmp_path, {"payload.txt"})


def test_repository_inventory_reader_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    """Restore must reject duplicate fields before inventory schema validation."""

    inventory = tmp_path / "repository.json"
    inventory.write_text(
        '{"schema_version":1,"schema_version":1}\n',
        encoding="utf-8",
    )
    config = portable.load_config_file(CONFIG_PATH)

    with pytest.raises(portable.PortableWorkspaceError, match="duplicate JSON key"):
        portable.load_repository_inventory(inventory, config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"created_at_utc": 123}),
        lambda value: value.update({"export_head_symbolic_ref": False}),
        lambda value: value.update({"changed_paths_after_source": 7}),
        lambda value: value.update({"archived_worktree_refs": "broken"}),
        lambda value: value.update({"worktrees": None}),
        lambda value: value.update({"bundle_heads": {"bad": True}}),
        lambda value: value.update({"bundle_verification": False}),
    ],
)
def test_repository_inventory_rejects_wrong_field_types(mutation) -> None:
    """Every restore-critical inventory field has an exact JSON type."""

    config, inventory = minimal_repository_inventory()
    portable.validate_repository_inventory(inventory, config)
    mutation(inventory)

    with pytest.raises(portable.PortableWorkspaceError):
        portable.validate_repository_inventory(inventory, config)


def test_restore_verifies_source_and_final_commit_trees(tmp_path: Path) -> None:
    """Both recorded commit trees must exist and match the fresh mirror."""

    repository = tmp_path / "repository"
    oid = initialize_repository(repository)
    tree = run_git(repository, "rev-parse", "HEAD^{tree}")
    inventory = {
        "source_base": oid,
        "source_base_tree": tree,
        "final_tip": oid,
        "final_tip_tree": tree,
    }
    assert portable.verify_inventory_commit_trees(repository, inventory) == {
        "source_base": tree,
        "final_tip": tree,
    }
    inventory["source_base_tree"] = "f" * 40

    with pytest.raises(portable.PortableWorkspaceError, match="source_base tree"):
        portable.verify_inventory_commit_trees(repository, inventory)


def test_main_restore_recomputes_symbolic_ref_chains() -> None:
    """Aliases that resolve through main must follow the promoted main object."""

    old_main = "1" * 40
    final_tip = "2" * 40
    main_ref = "refs/heads/main"
    alias_ref = "refs/custom/main-alias"
    refs = sorted(
        [
            {"refname": main_ref, "oid": old_main, "symref": ""},
            {"refname": alias_ref, "oid": old_main, "symref": main_ref},
        ],
        key=lambda item: item["refname"],
    )
    policy = {
        "local_main_ref": main_ref,
        "pre_migration_main_archive_ref": "refs/archive/pre-main",
    }

    restored = portable.expected_refs_after_main_restore(
        refs, final_tip, old_main, policy
    )
    objects = {item["refname"]: item["oid"] for item in restored}

    assert objects[main_ref] == final_tip
    assert objects[alias_ref] == final_tip
    assert objects[policy["pre_migration_main_archive_ref"]] == old_main


@pytest.mark.parametrize(
    "refs",
    [
        [
            {"refname": "refs/custom/a", "oid": "1" * 40, "symref": "refs/custom/b"},
        ],
        [
            {"refname": "refs/custom/a", "oid": "1" * 40, "symref": "refs/custom/b"},
            {"refname": "refs/custom/b", "oid": "1" * 40, "symref": "refs/custom/a"},
        ],
    ],
)
def test_symbolic_ref_graph_rejects_missing_targets_and_cycles(refs) -> None:
    """Restore must reject unresolved or cyclic symbolic-ref graphs."""

    with pytest.raises(portable.PortableWorkspaceError, match="symbolic ref"):
        portable.resolve_symbolic_ref_objects(refs)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/repository.git?private=value",
        "https://example.invalid/repository.git#private",
    ],
)
def test_remote_url_rejects_query_and_fragment(tmp_path: Path, url: str) -> None:
    """Only a public repository location is retained in the archive."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    run_git(repository, "remote", "add", "origin", url)

    with pytest.raises(portable.PortableWorkspaceError, match="query, or fragment"):
        portable.safe_remote_url(repository, "origin")


def test_bundle_round_trip_preserves_all_ref_names_objects_and_symrefs(
    tmp_path: Path,
) -> None:
    """A fresh mirror must reproduce branches, tags, custom refs, and symrefs."""

    repository = tmp_path / "repository"
    oid = initialize_repository(repository)
    run_git(repository, "branch", "experiment")
    run_git(repository, "tag", "v1")
    run_git(repository, "notes", "add", "-m", "portable note")
    run_git(repository, "update-ref", "refs/custom/keep", oid)
    run_git(repository, "update-ref", "refs/remotes/origin/main", oid)
    run_git(
        repository,
        "symbolic-ref",
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/main",
    )
    expected = portable.exact_refs(repository)
    bundle = tmp_path / "repository.bundle"
    portable.create_ref_inventory_bundle(repository, bundle, expected)
    portable.require_exact_bundle_heads(
        portable.bundle_heads(repository, bundle),
        expected,
        oid,
    )

    result = portable.verify_bundle_clone(repository, bundle, expected)

    assert result["fresh_mirror_exact_ref_match"] is True
    assert {
        item["refname"] for item in result["reconstructed_symbolic_refs"]
    } == {"refs/remotes/origin/HEAD"}


def test_ref_inventory_bundle_omits_linked_worktree_pseudo_heads(
    tmp_path: Path,
) -> None:
    """Bundle advertisement must exclude linked-worktree pseudo HEADs."""

    repository = tmp_path / "repository"
    oid = initialize_repository(repository)
    linked = tmp_path / "linked"
    run_git(repository, "worktree", "add", "-b", "linked", str(linked), "HEAD")
    expected = portable.exact_refs(repository)

    all_bundle = tmp_path / "all.bundle"
    run_git(repository, "bundle", "create", str(all_bundle), "--all")
    with pytest.raises(
        portable.PortableWorkspaceError,
        match="bundle advertised heads differ from ref inventory",
    ):
        portable.require_exact_bundle_heads(
            portable.bundle_heads(repository, all_bundle), expected, oid
        )

    exact_bundle = tmp_path / "exact.bundle"
    portable.create_ref_inventory_bundle(repository, exact_bundle, expected)
    exact_heads = portable.bundle_heads(repository, exact_bundle)
    portable.require_exact_bundle_heads(exact_heads, expected, oid)
    assert {head["refname"] for head in exact_heads} == {
        "HEAD",
        *(ref["refname"] for ref in expected),
    }


def test_repository_preflight_rejects_shallow_and_submodule_sources(
    tmp_path: Path,
) -> None:
    """A bundle source must own complete objects and contain no gitlink entries."""

    shallow = tmp_path / "shallow"
    oid = initialize_repository(shallow)
    git_directory = Path(run_git(shallow, "rev-parse", "--git-dir"))
    if not git_directory.is_absolute():
        git_directory = shallow / git_directory
    (git_directory / "shallow").write_text(oid + "\n", encoding="ascii")
    with pytest.raises(portable.PortableWorkspaceError, match="shallow"):
        portable.assert_repository_self_contained(shallow, oid)

    repository = tmp_path / "submodule-source"
    oid = initialize_repository(repository)
    run_git(repository, "branch", "legacy")
    run_git(repository, "switch", "legacy")
    run_git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{oid},nested-repository",
    )
    run_git(repository, "commit", "-m", "add gitlink")
    run_git(repository, "switch", "main")
    tip = run_git(repository, "rev-parse", "HEAD")
    with pytest.raises(portable.PortableWorkspaceError, match="submodules"):
        portable.assert_repository_self_contained(repository, tip)


def test_bootstrap_copies_final_blob_from_clean_smudged_file(tmp_path: Path) -> None:
    """A standard Git smudge cannot alter the exact archived bootstrap bytes."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    (repository / ".gitattributes").write_bytes(b"bootstrap.txt ident\n")
    run_git(repository, "add", ".gitattributes")
    run_git(repository, "commit", "-m", "attributes")
    source = repository / "bootstrap.txt"
    source.write_bytes(b"$Id$\n")
    run_git(repository, "add", "bootstrap.txt")
    run_git(repository, "commit", "-m", "bootstrap")
    tip = run_git(repository, "rev-parse", "HEAD")
    source.unlink()
    run_git(repository, "checkout", "--", "bootstrap.txt")

    assert run_git(repository, "status", "--porcelain") == ""
    assert source.read_bytes() != portable.tracked_blob_at_tip(
        repository, "bootstrap.txt", tip
    )
    assert portable.require_worktree_file_matches_tip(
        repository, "bootstrap.txt", tip
    ) == portable.tracked_blob_at_tip(repository, "bootstrap.txt", tip)

    source.write_bytes(b"different\n")
    with pytest.raises(portable.PortableWorkspaceError, match="differs from final"):
        portable.require_worktree_file_matches_tip(repository, "bootstrap.txt", tip)


def test_worktree_preflight_rejects_private_refs_and_operations(
    tmp_path: Path,
) -> None:
    """Linked-worktree private refs and unfinished Git operations must block export."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    linked = tmp_path / "linked"
    run_git(repository, "worktree", "add", "-b", "linked", str(linked), "HEAD")
    records = portable.parse_worktree_records(repository)
    run_git(linked, "update-ref", "refs/bisect/good", "HEAD")
    with pytest.raises(portable.PortableWorkspaceError, match="worktree-private refs"):
        portable.assert_no_worktree_private_git_state(repository, records)

    run_git(linked, "update-ref", "-d", "refs/bisect/good")
    git_directory = Path(run_git(linked, "rev-parse", "--git-dir"))
    if not git_directory.is_absolute():
        git_directory = linked / git_directory
    (git_directory / "MERGE_HEAD").write_text("1" * 40 + "\n", encoding="ascii")
    with pytest.raises(portable.PortableWorkspaceError, match="in-progress"):
        portable.assert_no_worktree_private_git_state(repository, records)


def test_unreachable_detached_worktree_receives_archival_ref(
    tmp_path: Path,
) -> None:
    """A detached worktree commit must become reachable before bundle creation."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    worktree = tmp_path / "detached"
    run_git(repository, "worktree", "add", "--detach", str(worktree), "HEAD")
    (worktree / "detached.txt").write_text("preserve\n", encoding="utf-8")
    run_git(worktree, "add", "detached.txt")
    run_git(worktree, "commit", "-m", "detached work")
    detached_oid = run_git(worktree, "rev-parse", "HEAD")
    records = portable.parse_worktree_records(repository)
    ordinal = next(
        index
        for index, record in enumerate(records)
        if Path(record["path"]) == worktree
    )

    assert portable.named_refs_containing(repository, detached_oid) == []
    archived = portable.ensure_worktree_heads_reachable(
        repository, records, "refs/archive/worktrees"
    )

    assert archived == [
        {
            "refname": f"refs/archive/worktrees/{ordinal:03d}-{detached_oid[:12]}",
            "oid": detached_oid,
        }
    ]
    assert portable.named_refs_containing(repository, detached_oid) == [
        f"refs/archive/worktrees/{ordinal:03d}-{detached_oid[:12]}"
    ]


def test_dirty_crlf_snapshot_preserves_raw_and_filtered_identities(
    tmp_path: Path,
) -> None:
    """Dirty snapshots must retain exact bytes and Git-normalized identities."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    (repository / ".gitattributes").write_text("*.py text eol=lf\n", encoding="utf-8")
    source = repository / "sample.py"
    source.write_bytes(b"value = 1\n")
    run_git(repository, "add", ".gitattributes", "sample.py")
    run_git(repository, "commit", "-m", "tracked source")
    source.write_bytes(b"value = 2\r\n")
    staging = tmp_path / "staging"
    staging.mkdir()

    config = portable.load_config_file(CONFIG_PATH)
    records = portable.collect_worktrees(
        repository,
        staging,
        1024 * 1024,
        repository,
        config.archive,
        config.context,
    )
    record = next(item for item in records if Path(item["path"]) == repository)
    file_record = next(
        item for item in record["snapshot_files"] if item["path"] == "sample.py"
    )
    archived = (
        staging
        / record["snapshot_path"]
        / Path(file_record["archive_path"])
    )

    assert archived.read_bytes() == b"value = 2\r\n"
    assert file_record["sha256"] == hashlib.sha256(b"value = 2\r\n").hexdigest()
    assert file_record["raw_blob_oid"] != file_record["filtered_blob_oid"]
    assert file_record["head_blob_oid"] in file_record["index_records"][0]


def test_worktree_inventory_rejects_unknown_ignored_files(tmp_path: Path) -> None:
    """Git ignore rules cannot silently remove an unclassified local file."""

    repository = tmp_path / "repository"
    initialize_repository(repository)
    (repository / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    run_git(repository, "add", ".gitignore")
    run_git(repository, "commit", "-m", "ignore local file")
    (repository / "ignored.txt").write_text("important\n", encoding="utf-8")
    config = portable.load_config_file(CONFIG_PATH)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(portable.PortableWorkspaceError, match="lack migration policy"):
        portable.collect_worktrees(
            repository,
            staging,
            1024,
            repository,
            config.archive,
            config.context,
        )


def test_powershell_wrappers_parse() -> None:
    """Windows PowerShell must parse both thin entry points without execution."""

    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    for name in (
        "export_portable_workspace.ps1",
        "restore_portable_workspace.ps1",
    ):
        path = MIGRATION_DIRECTORY / name
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-Command",
                f"[ScriptBlock]::Create((Get-Content -Raw -LiteralPath '{path}')) | Out-Null",
            ],
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode(
            "utf-8", errors="replace"
        )
