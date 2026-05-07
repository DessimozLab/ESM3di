#!/usr/bin/env python
"""Utilities for publishing ESM3Di artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def _resolve_hf_token(token_env_var: str) -> str | None:
    """Resolve token from environment if available."""
    return os.getenv(token_env_var) or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _require_hf_token(token_env_var: str) -> str:
    """Return HF token or raise a clear configuration error."""
    token = _resolve_hf_token(token_env_var)
    if token:
        return token
    raise RuntimeError(
        "No Hugging Face token found. Set one of these environment variables: "
        f"{token_env_var}, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN."
    )


def publish_directory_to_hub(
    local_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    revision: str = "main",
    commit_message: str = "Upload model artifacts",
    token_env_var: str = "HF_TOKEN",
) -> str:
    """Upload a local directory to a Hugging Face model repository."""
    local_dir = Path(local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {local_dir}")

    token = _require_hf_token(token_env_var)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def publish_file_to_hub(
    local_file: str | Path,
    repo_id: str,
    *,
    path_in_repo: str,
    private: bool = False,
    revision: str = "main",
    commit_message: str = "Upload model artifact",
    token_env_var: str = "HF_TOKEN",
) -> str:
    """Upload a single file to a Hugging Face model repository."""
    local_file = Path(local_file)
    if not local_file.exists() or not local_file.is_file():
        raise FileNotFoundError(f"File does not exist: {local_file}")

    token = _require_hf_token(token_env_var)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish ESM3Di artifacts (directories or files) to Hugging Face Hub.",
    )
    parser.add_argument("--repo-id", required=True, help="Target model repository id, e.g. org/esm3di-model")
    parser.add_argument("--local-dir", default=None, help="Local directory to upload with upload_folder")
    parser.add_argument("--local-file", default=None, help="Local file to upload with upload_file")
    parser.add_argument("--path-in-repo", default=None, help="Required when --local-file is provided")
    parser.add_argument("--private", action="store_true", help="Create private repository if it does not exist")
    parser.add_argument("--revision", default="main", help="Target revision/branch in HF repo")
    parser.add_argument("--commit-message", default="Upload model artifacts", help="Commit message for upload")
    parser.add_argument("--token-env-var", default="HF_TOKEN", help="Env var used to read HF token")
    args = parser.parse_args()

    has_dir = bool(args.local_dir)
    has_file = bool(args.local_file)
    if has_dir == has_file:
        parser.error("Provide exactly one of --local-dir or --local-file")

    if has_file and not args.path_in_repo:
        parser.error("--path-in-repo is required when using --local-file")

    if has_dir:
        repo_url = publish_directory_to_hub(
            local_dir=args.local_dir,
            repo_id=args.repo_id,
            private=args.private,
            revision=args.revision,
            commit_message=args.commit_message,
            token_env_var=args.token_env_var,
        )
    else:
        repo_url = publish_file_to_hub(
            local_file=args.local_file,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            private=args.private,
            revision=args.revision,
            commit_message=args.commit_message,
            token_env_var=args.token_env_var,
        )

    print(f"Published to {repo_url}")


if __name__ == "__main__":
    main()
