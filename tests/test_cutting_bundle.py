import pytest
import torch

from esm3di.inference import cutting, cutting_bundle


def sources(tmp_path):
    embedding = tmp_path / "embedding.pt"
    torch.save({
        "args": {"hf_model": "Synthyra/ESMplusplus_small", "aa_fasta": "/private/training.fa"},
        "label_vocab": list("AC"),
        "model_state_dict": {"weight": torch.arange(6).reshape(2, 3).float()},
        "optimizer_state_dict": {"not_exported": True},
    }, embedding)
    paths = [embedding]
    for name in ("ffn", "transformer"):
        path = tmp_path / f"{name}.pt"
        torch.save({
            "state_dict": {"head.weight": torch.ones(1, 3)},
            "d_in": 3, "model_type": name, "esm3di_checkpoint": str(embedding),
        }, path)
        paths.append(path)
    return paths


def test_export_preserves_tensors_without_models_or_inference(tmp_path):
    paths = sources(tmp_path)
    output = tmp_path / "bundle"
    cutting_bundle.export_bundle(*paths, output)
    config = cutting_bundle.read_bundle(output)
    assert config["embedding"]["args"]["revision"]
    assert "/private" not in (output / "cutting_config.json").read_text()
    assert "optimizer_state_dict" not in config["embedding"]
    for name, path in zip(("embedding", "ffn", "transformer"), paths):
        original = torch.load(path, weights_only=False)
        restored = cutting_bundle.load_bundle_checkpoint(output, name)
        key = "model_state_dict" if name == "embedding" else "state_dict"
        assert original[key].keys() == restored[key].keys()
        for tensor in original[key]:
            assert torch.equal(original[key][tensor], restored[key][tensor])
    with pytest.raises(FileExistsError):
        cutting_bundle.export_bundle(*paths, output)


def test_checksum_failure_is_explicit(tmp_path):
    output = cutting_bundle.export_bundle(*sources(tmp_path), tmp_path / "bundle")
    (output / "ffn.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        cutting_bundle.read_bundle(output)


def test_hub_resolution_uses_separate_subfolder_and_revision(tmp_path, monkeypatch):
    bundle = tmp_path / "snapshot/hf_compatible/cutter"
    cutting_bundle.export_bundle(*sources(tmp_path), bundle)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(cutting_bundle, "snapshot_download", download)
    assert cutting_bundle.resolve_bundle("owner/repo", "hf_compatible/cutter", "commit") == bundle
    assert calls == [{
        "repo_id": "owner/repo", "revision": "commit",
        "allow_patterns": ["hf_compatible/cutter/*"],
    }]


def test_bundle_cli_routes_without_inference(tmp_path, monkeypatch):
    bundle = cutting_bundle.export_bundle(*sources(tmp_path), tmp_path / "bundle")
    calls = []
    monkeypatch.setattr(cutting, "run_cutting_pipeline", lambda **kwargs: calls.append(kwargs))
    cutting.main([
        "--input-fasta", "input.fa", "--cutter-model", str(bundle),
        "--output-prefix", "out",
    ])
    assert calls[0]["esm3di_checkpoint"] == str(bundle)
    assert calls[0]["ffn_checkpoint"] == str(bundle)
    assert calls[0]["transformer_checkpoint"] == str(bundle)


def test_export_refuses_mismatched_embedding_provenance(tmp_path):
    paths = sources(tmp_path)
    checkpoint = torch.load(paths[1], weights_only=False)
    checkpoint["esm3di_checkpoint"] = "different.pt"
    torch.save(checkpoint, paths[1])
    with pytest.raises(ValueError, match="different embedding"):
        cutting_bundle.export_bundle(*paths, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_heads_only_export_references_existing_model(tmp_path):
    output = cutting_bundle.export_bundle(
        *sources(tmp_path), tmp_path / "heads",
        embedding_repo="owner/repo", embedding_revision="pinned-commit",
    )
    config = cutting_bundle.read_bundle(output)
    assert config["format"] == "esm3di-cutting-v2"
    assert config["embedding"]["repo_revision"] == "pinned-commit"
    assert not (output / "embedding.safetensors").exists()
    assert len(list(output.iterdir())) == 4


def test_heads_only_loader_uses_pinned_existing_hub_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    from esm3di.inference import predictor

    output = cutting_bundle.export_bundle(
        *sources(tmp_path), tmp_path / "heads",
        embedding_repo="owner/repo", embedding_revision="pinned-commit",
    )
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(
        predictor, "ESM3DiPredictor",
        lambda *args, **kwargs: SimpleNamespace(model="existing-model", tokenizer="tokenizer"),
    )
    assert cutting.load_esm3di_embedding_model(str(output), "cpu") == (
        "existing-model", "tokenizer", False,
    )
    assert calls[0]["revision"] == "pinned-commit"
