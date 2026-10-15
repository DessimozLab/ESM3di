from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    AutoModelForMaskedLM, AutoModelForTokenClassification,
    BertConfig, BertForMaskedLM, BertForTokenClassification, PreTrainedTokenizerFast,
)

from esm3di._core import checkpoints
from esm3di_training import train_3di, train_mlm


@pytest.fixture
def tiny_backbone(monkeypatch):
    """Exercise real optimizers, transfer, PEFT, CNN and serialization offline."""
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *"ACDEFGHIKLMNPQRSTVWY"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )

    def config():
        return BertConfig(
            vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, num_labels=20,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            tie_word_embeddings=False,
        )

    def create(model_class):
        model = model_class(config())
        model.tokenizer = tokenizer
        return model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", lambda *a, **kw: create(BertForMaskedLM))
    monkeypatch.setattr(AutoModelForTokenClassification, "from_pretrained", lambda *a, **kw: create(BertForTokenClassification))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield tokenizer
    torch.set_num_threads(threads)


def test_mlm_to_supervised_training_roundtrip(tmp_path, tiny_backbone):
    aa = tmp_path / "aa.fasta"
    labels = tmp_path / "3di.fasta"
    bins = tmp_path / "bins.fasta"
    aa.write_text(">one\nACDEFG\n>two\nHIKLMN\n")
    labels.write_text(">one\nACDEFG\n>two\nHIKLMN\n")
    bins.write_text(">one\n999999\n>two\n777777\n")
    mlm_cfg = dict(train_mlm.DEFAULTS, input_fasta=[str(aa)], val_fasta=[str(aa)],
                   output_dir=str(tmp_path / "mlm"), epochs=1, batch_size=2,
                   grad_accum_steps=1, num_workers=0, device="cpu", precision="fp32",
                   mlm_probability=1.0, log_every=1)
    mlm_best = train_mlm.train(mlm_cfg)
    assert Path(mlm_best, "model.safetensors").is_file()
    source = checkpoints.load_mlm_state_dict(mlm_best)
    continued, _, report = checkpoints.load_mlm_model(mlm_best)
    assert report.copied > 0
    assert torch.equal(continued.bert.embeddings.word_embeddings.weight, source["bert.embeddings.word_embeddings.weight"])

    cfg = dict(train_3di.DEFAULTS, aa_fasta=str(aa), three_di_fasta=str(labels),
               plddt_bins_fasta=str(bins), val_aa_fasta=str(aa),
               val_three_di_fasta=str(labels), val_plddt_bins_fasta=str(bins),
               init_from=mlm_best, out_dir=str(tmp_path / "3di"),
               epochs=1, batch_size=2, gradient_accumulation_steps=1,
               num_workers=0, device="cpu", mixed_precision=False,
               lora_r=2, lora_target_modules=["query", "value"], log_every=1)
    esm, _, _ = train_3di.build_model(cfg, None)
    assert esm.backbone_transfer_report.missing == []
    assert torch.equal(esm.base_model.bert.embeddings.word_embeddings.weight, source["bert.embeddings.word_embeddings.weight"])
    best = train_3di.train(cfg)
    saved = checkpoints.read_3di_checkpoint(best)
    assert saved["epoch"] == 1 and saved["global_step"] == 1
    assert saved["optimizer_state_dict"]["state"]
    assert saved["label_vocab"] == checkpoints.THREEDI_VOCAB
    assert not any(saved["args"].get(flag) for flag in checkpoints._UNSUPPORTED_LEGACY_FLAGS)

    rebuilt, _ = checkpoints.load_3di_checkpoint(best, device="cpu")
    second, _ = checkpoints.load_3di_checkpoint(str(tmp_path / "3di" / "last.pt"), device="cpu")
    encoded = tiny_backbone(["ACDEFG"], return_tensors="pt")
    with torch.inference_mode():
        first_logits = rebuilt.model(**encoded).logits
        last_logits = second.model(**encoded).logits
    assert torch.equal(first_logits, last_logits)
    assert first_logits.shape == (1, 8, 20)
    assert torch.equal(rebuilt.base_model.bert.embeddings.word_embeddings.weight, source["bert.embeddings.word_embeddings.weight"])
    output = tmp_path / "predicted.fasta"
    predicted = rebuilt.predict_from_fasta(str(aa), str(output), device="cpu")
    assert output.read_text() == "".join(f">{record['header']}\n{record['pred_3di']}\n" for record in predicted)
    assert [len(record["pred_3di"]) for record in predicted] == [6, 6]

    # Fresh runs refuse to replace either stage's saved artifacts.
    before = Path(best).read_bytes()
    with pytest.raises(FileExistsError):
        train_3di.train(cfg)
    with pytest.raises(FileExistsError):
        train_mlm.train(mlm_cfg)
    assert Path(best).read_bytes() == before


@pytest.mark.parametrize("flag", checkpoints._UNSUPPORTED_LEGACY_FLAGS)
def test_archived_heads_rejected_explicitly(flag):
    with pytest.raises(ValueError, match=flag):
        checkpoints.model_kwargs_from_checkpoint({"args": {flag: True}, "label_vocab": list("ABCDEFGHIJKLMNOPQRST")})


def test_esm2_checkpoint_rejected_explicitly():
    with pytest.raises(ValueError, match="Only ESM\\+\\+"):
        checkpoints.model_kwargs_from_checkpoint({
            "args": {"hf_model": "facebook/esm2_t12_35M_UR50D", "use_cnn_head": True},
            "label_vocab": list("ABCDEFGHIJKLMNOPQRST"),
        })


def test_esm2_mlm_rejected_before_download():
    with pytest.raises(ValueError, match="Only ESM\\+\\+ MLM"):
        checkpoints.load_mlm_model("facebook/esm2_t12_35M_UR50D")
