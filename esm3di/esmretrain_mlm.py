#!/usr/bin/env python
import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
	AutoModelForMaskedLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .ESM3di_model import read_fasta, discover_lora_target_modules


@dataclass
class TrainStats:
	loss_sum: float = 0.0
	steps: int = 0

	def update(self, loss_value: float):
		self.loss_sum += loss_value
		self.steps += 1

	def average(self) -> float:
		return self.loss_sum / max(self.steps, 1)


class SequenceDataset(Dataset):
	def __init__(self, sequences: List[str]):
		self.sequences = sequences

	def __len__(self) -> int:
		return len(self.sequences)

	def __getitem__(self, idx: int):
		return {"sequence": self.sequences[idx]}


def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_sequences_from_fasta(fasta_path: str, min_length: int = 1) -> List[str]:
	records = read_fasta(fasta_path)
	sequences = []
	for _, seq in records:
		seq = seq.strip().upper()
		if len(seq) >= min_length:
			sequences.append(seq)
	return sequences


def load_sequences_from_txt(txt_path: str, min_length: int = 1) -> List[str]:
	sequences = []
	with open(txt_path, "r") as handle:
		for line in handle:
			seq = line.strip().upper()
			if not seq:
				continue
			if len(seq) >= min_length:
				sequences.append(seq)
	return sequences


def resolve_dtype(dtype_str: str):
	if dtype_str == "fp16":
		return torch.float16
	if dtype_str == "bf16":
		return torch.bfloat16
	return torch.float32


def load_tokenizer_with_fallback(model_name: str):
	try:
		return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	except ValueError as exc:
		msg = str(exc)
		if "EsmSequenceTokenizer" not in msg:
			raise
		# Fallback for ESM++ repos that expose tokenizer via the model object.
		model = AutoModelForMaskedLM.from_pretrained(
			model_name,
			trust_remote_code=True,
		)
		tokenizer = getattr(model, "tokenizer", None)
		if tokenizer is None:
			raise
		return tokenizer


def make_collate_fn(tokenizer, mlm_probability: float, max_length: Optional[int]):
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm_probability=mlm_probability,
	)

	def _normalize_encodings(encodings):
		if isinstance(encodings, dict) and "input_ids" in encodings:
			return encodings
		if hasattr(encodings, "encodings"):
			enc_list = encodings.encodings
		elif isinstance(encodings, list):
			enc_list = encodings
		elif hasattr(encodings, "ids"):
			enc_list = [encodings]
		else:
			raise ValueError("Unsupported tokenizer output type for MLM collate.")

		pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
		input_ids = []
		attention_masks = []
		for enc in enc_list:
			ids = list(getattr(enc, "ids", []))
			if not ids and isinstance(enc, dict) and "input_ids" in enc:
				ids = list(enc["input_ids"])
			if max_length is not None:
				ids = ids[:max_length]
			attn = list(getattr(enc, "attention_mask", [1] * len(ids)))
			if max_length is not None:
				attn = attn[:max_length]
			input_ids.append(ids)
			attention_masks.append(attn)

		pad_len = max(len(ids) for ids in input_ids) if input_ids else 0
		input_ids = [
			ids + [pad_id] * (pad_len - len(ids))
			for ids in input_ids
		]
		attention_masks = [
			mask + [0] * (pad_len - len(mask))
			for mask in attention_masks
		]

		return {
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"attention_mask": torch.tensor(attention_masks, dtype=torch.long),
		}

	def collate(batch):
		sequences = [item["sequence"] for item in batch]
		try:
			tokenized = tokenizer(
				sequences,
				padding=True,
				truncation=True,
				max_length=max_length,
				return_tensors="pt",
			)
		except Exception:
			tokenized = tokenizer(
				sequences,
				padding=True,
				truncation=True,
				max_length=max_length,
			)
		if not (isinstance(tokenized, dict) and "input_ids" in tokenized):
			tokenized = _normalize_encodings(tokenized)
		if isinstance(tokenized, dict):
			batch_size = tokenized["input_ids"].size(0)
			features = [
				{k: v[i] for k, v in tokenized.items()}
				for i in range(batch_size)
			]
			return data_collator(features)
		return data_collator(tokenized)

	return collate


def build_loaders(
	sequences: List[str],
	tokenizer,
	mlm_probability: float,
	max_length: Optional[int],
	batch_size: int,
	val_split: float,
	seed: int,
	num_workers: int,
):
	dataset = SequenceDataset(sequences)
	if val_split <= 0:
		train_dataset = dataset
		val_dataset = None
	else:
		val_size = max(1, int(len(dataset) * val_split))
		train_size = max(1, len(dataset) - val_size)
		generator = torch.Generator().manual_seed(seed)
		train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

	collate_fn = make_collate_fn(tokenizer, mlm_probability, max_length)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		collate_fn=collate_fn,
	)
	val_loader = None
	if val_dataset is not None:
		val_loader = DataLoader(
			val_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=torch.cuda.is_available(),
			collate_fn=collate_fn,
		)

	return train_loader, val_loader


def build_eval_loader(
	sequences: List[str],
	tokenizer,
	mlm_probability: float,
	max_length: Optional[int],
	batch_size: int,
	num_workers: int,
):
	dataset = SequenceDataset(sequences)
	collate_fn = make_collate_fn(tokenizer, mlm_probability, max_length)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		collate_fn=collate_fn,
	)


def train_one_epoch(
	model,
	train_loader,
	optimizer,
	scheduler,
	device,
	use_amp: bool,
	grad_accum_steps: int,
	scaler: Optional[torch.cuda.amp.GradScaler],
	logger: logging.Logger,
	log_every: int,
	epoch: int,
):
	model.train()
	stats = TrainStats()
	optimizer.zero_grad(set_to_none=True)

	progress = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
	for step, batch in enumerate(progress):
		batch = {k: v.to(device) for k, v in batch.items()}
		with torch.cuda.amp.autocast(enabled=use_amp):
			outputs = model(**batch)
			loss = outputs.loss
			loss = loss / max(grad_accum_steps, 1)

		if scaler is not None:
			scaler.scale(loss).backward()
		else:
			loss.backward()

		if (step + 1) % grad_accum_steps == 0:
			if scaler is not None:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			optimizer.zero_grad(set_to_none=True)
			if scheduler is not None:
				scheduler.step()

		stats.update(loss.item() * max(grad_accum_steps, 1))
		if log_every > 0 and (step + 1) % log_every == 0:
			logger.info(
				"epoch=%d step=%d loss=%.6f",
				epoch,
				step + 1,
				stats.average(),
			)
			progress.set_postfix(loss=f"{stats.average():.6f}")

	return stats.average()


@torch.no_grad()
def validate(model, val_loader, device, use_amp: bool, epoch: int):
	model.eval()
	stats = TrainStats()
	progress = tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False)
	for batch in progress:
		batch = {k: v.to(device) for k, v in batch.items()}
		with torch.cuda.amp.autocast(enabled=use_amp):
			outputs = model(**batch)
			loss = outputs.loss
		stats.update(loss.item())
		progress.set_postfix(loss=f"{stats.average():.6f}")
	return stats.average()


def _is_peft_model(model) -> bool:
	return hasattr(model, "peft_config")


def save_checkpoint(model, tokenizer, output_dir: Path, tag: str):
	target_dir = output_dir / tag
	target_dir.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(target_dir)
	tokenizer.save_pretrained(target_dir)


def parse_args():
	parser = argparse.ArgumentParser(description="Masked language model retraining for protein sequences.")
	parser.add_argument("--input-fasta", type=str, help="Path to input FASTA with AA sequences.")
	parser.add_argument("--input-txt", type=str, help="Path to input text with one sequence per line.")
	parser.add_argument("--val-fasta", type=str, help="Optional FASTA for validation sequences.")
	parser.add_argument("--test-fasta", type=str, help="Optional FASTA for test sequences.")
	parser.add_argument("--model-name", type=str, required=True, help="HF model name (ESMC/ESM++).")
	parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints.")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--learning-rate", type=float, default=2e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-steps", type=int, default=0)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--mlm-probability", type=float, default=0.15)
	parser.add_argument("--val-split", type=float, default=0.05)
	parser.add_argument("--grad-accum-steps", type=int, default=1)
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--min-length", type=int, default=1)
	parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp32")
	parser.add_argument("--config", type=str, default=None, help="Optional JSON config to override args.")
	parser.add_argument("--log-every", type=int, default=100, help="Log every N optimizer steps.")
	parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters during MLM retraining.")
	parser.add_argument("--lora-r", type=int, default=8)
	parser.add_argument("--lora-alpha", type=float, default=16.0)
	parser.add_argument("--lora-dropout", type=float, default=0.05)
	parser.add_argument("--lora-targets", type=str, default=None, help="Comma-separated LoRA target module names.")
	parser.add_argument(
		"--merge-lora-on-save",
		action="store_true",
		help="Merge LoRA weights into base model at the end and save a full ESMC checkpoint.",
	)
	return parser.parse_args()


def load_config_if_present(args):
	if not args.config:
		return args
	with open(args.config, "r") as handle:
		config = json.load(handle)
	for key, value in config.items():
		if hasattr(args, key):
			setattr(args, key, value)
	return args


def find_latest_epoch_checkpoint(output_dir: Path) -> Optional[int]:
	if not output_dir.exists() or not output_dir.is_dir():
		return None
	epochs = []
	for entry in output_dir.iterdir():
		if not entry.is_dir():
			continue
		name = entry.name
		if name.startswith("epoch_"):
			try:
				epochs.append(int(name.split("epoch_")[-1]))
			except ValueError:
				continue
	if not epochs:
		return None
	return max(epochs)


def load_model_checkpoint(model, checkpoint_dir: Path, args, tokenizer, dtype):
	# Load base model + tokenizer and optionally LoRA adapters.
	if not checkpoint_dir.exists():
		return model

	model = AutoModelForMaskedLM.from_pretrained(
		checkpoint_dir,
		trust_remote_code=True,
		torch_dtype=dtype,
	)

	if args.use_lora:
		# Put base model into PEFT wrapper with existing weights:
		model = PeftModel.from_pretrained(model, checkpoint_dir)
	else:
		if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
			model.config.pad_token_id = tokenizer.pad_token_id
	if len(tokenizer) != model.get_input_embeddings().num_embeddings:
		model.resize_token_embeddings(len(tokenizer))
	return model


def main():
	args = load_config_if_present(parse_args())

	if not args.input_fasta and not args.input_txt:
		raise ValueError("Provide --input-fasta or --input-txt with sequences.")

	set_seed(args.seed)

	sequences: List[str] = []
	if args.input_fasta:
		sequences.extend(load_sequences_from_fasta(args.input_fasta, min_length=args.min_length))
	if args.input_txt:
		sequences.extend(load_sequences_from_txt(args.input_txt, min_length=args.min_length))

	val_sequences: List[str] = []
	if args.val_fasta:
		val_sequences.extend(load_sequences_from_fasta(args.val_fasta, min_length=args.min_length))

	test_sequences: List[str] = []
	if args.test_fasta:
		test_sequences.extend(load_sequences_from_fasta(args.test_fasta, min_length=args.min_length))

	if not sequences:
		raise ValueError("No sequences loaded. Check inputs or min-length filter.")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	logger = logging.getLogger("esm3di_mlm")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
	logger.addHandler(console_handler)
	file_handler = logging.FileHandler(output_dir / "train.log")
	file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
	logger.addHandler(file_handler)

	dtype = resolve_dtype(args.dtype)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"

	tokenizer = load_tokenizer_with_fallback(args.model_name)
	if tokenizer.pad_token is None:
		if tokenizer.eos_token is not None:
			tokenizer.pad_token = tokenizer.eos_token
		elif tokenizer.cls_token is not None:
			tokenizer.pad_token = tokenizer.cls_token
		else:
			tokenizer.add_special_tokens({"pad_token": "[PAD]"})

	# If there are saved epochs, resume from latest checkpoint
	resumed_epoch = find_latest_epoch_checkpoint(output_dir) or 0
	if resumed_epoch > 0:
		logger.info("Resuming from existing checkpoint at epoch %d", resumed_epoch)
		model = load_model_checkpoint(None, output_dir / f"epoch_{resumed_epoch}", args, tokenizer, dtype)
		start_epoch = resumed_epoch + 1
	else:
		model = AutoModelForMaskedLM.from_pretrained(
			args.model_name,
			trust_remote_code=True,
			torch_dtype=dtype,
		)
		if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
			model.config.pad_token_id = tokenizer.pad_token_id
		if len(tokenizer) != model.get_input_embeddings().num_embeddings:
			model.resize_token_embeddings(len(tokenizer))

		if args.use_lora:
			if args.lora_targets:
				target_modules = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
			else:
				target_modules = discover_lora_target_modules(model)
			lora_config = LoraConfig(
				r=args.lora_r,
				lora_alpha=args.lora_alpha,
				lora_dropout=args.lora_dropout,
				bias="none",
				task_type=TaskType.MASKED_LM,
				target_modules=target_modules,
			)
			model = get_peft_model(model, lora_config)
			model.print_trainable_parameters()

		start_epoch = 1

	model.to(device)

	if val_sequences:
		train_loader = build_eval_loader(
			sequences,
			tokenizer,
			args.mlm_probability,
			args.max_length,
			args.batch_size,
			args.num_workers,
		)
		val_loader = build_eval_loader(
			val_sequences,
			tokenizer,
			args.mlm_probability,
			args.max_length,
			args.batch_size,
			args.num_workers,
		)
	else:
		train_loader, val_loader = build_loaders(
			sequences,
			tokenizer,
			args.mlm_probability,
			args.max_length,
			args.batch_size,
			args.val_split,
			args.seed,
			args.num_workers,
		)

	test_loader = None
	if test_sequences:
		test_loader = build_eval_loader(
			test_sequences,
			tokenizer,
			args.mlm_probability,
			args.max_length,
			args.batch_size,
			args.num_workers,
		)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)

	total_steps = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)) * max(0, (args.epochs - start_epoch + 1))
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=args.warmup_steps,
		num_training_steps=total_steps,
	)

	scaler = torch.cuda.amp.GradScaler(enabled=use_amp and dtype == torch.float16)

	best_val = None
	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	logger.info("Loaded %d sequences | run_id=%s", len(sequences), run_id)


	if start_epoch > args.epochs:
		logger.info("No training needed: already at or beyond target epoch %d", args.epochs)
		train_loss = None
		val_loss = None
	else:
		for epoch in range(start_epoch, args.epochs + 1):
			train_loss = train_one_epoch(
				model,
				train_loader,
				optimizer,
				scheduler,
				device,
				use_amp=use_amp,
				grad_accum_steps=args.grad_accum_steps,
				scaler=scaler,
				logger=logger,
				log_every=args.log_every,
				epoch=epoch,
			)

		val_loss = None
		if val_loader is not None:
			val_loss = validate(model, val_loader, device, use_amp=use_amp, epoch=epoch)

		epoch_tag = f"epoch_{epoch}"
		save_checkpoint(model, tokenizer, output_dir, epoch_tag)

		if val_loss is not None:
			if best_val is None or val_loss < best_val:
				best_val = val_loss
				save_checkpoint(model, tokenizer, output_dir, "best")

		if val_loss is None:
			logger.info("Epoch %d: train_loss=%.6f", epoch, train_loss)
		else:
			logger.info("Epoch %d: train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

	if start_epoch > args.epochs:
		last_epoch = resumed_epoch
	else:
		last_epoch = args.epochs

	# Ensure testing uses the last saved checkpoint (resumed or newly trained)
	last_checkpoint_dir = output_dir / f"epoch_{last_epoch}"
	if last_checkpoint_dir.exists():
		logger.info("Loading checkpoint from epoch %d for final testing", last_epoch)
		model = load_model_checkpoint(model, last_checkpoint_dir, args, tokenizer, dtype)
		model.to(device)

	save_checkpoint(model, tokenizer, output_dir, "final")

	# Save model in Hugging Face compatible format for easy loading and sharing
	hf_output_dir = output_dir / "hf_compatible"
	hf_output_dir.mkdir(parents=True, exist_ok=True)
	hf_model = model.module if hasattr(model, "module") else model
	try:
		if hasattr(hf_model, "save_pretrained"):
			hf_model.save_pretrained(hf_output_dir)
		else:
			backend = getattr(hf_model, "base_model", None)
			if backend is not None and hasattr(backend, "save_pretrained"):
				backend.save_pretrained(hf_output_dir)
			else:
				logger.warning("Could not find save_pretrained() on model or base_model, HF compatible save skipped")
		if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
			tokenizer.save_pretrained(hf_output_dir)
		logger.info("Saved Hugging Face compatible model to %s", hf_output_dir)
	except Exception as exc:
		logger.warning("Failed to export HF compatible model: %s", exc)

	if test_loader is not None:
		test_loss = validate(model, test_loader, device, use_amp=use_amp, epoch=last_epoch)
		logger.info("Final test_loss=% .6f", test_loss)

	if args.use_lora:
		adapter_dir = output_dir / "lora_adapters"
		adapter_dir.mkdir(parents=True, exist_ok=True)
		model.save_pretrained(adapter_dir)
		if args.merge_lora_on_save and hasattr(model, "merge_and_unload"):
			merged_model = model.merge_and_unload()
			merged_dir = output_dir / "merged"
			merged_dir.mkdir(parents=True, exist_ok=True)
			merged_model.save_pretrained(merged_dir)
			tokenizer.save_pretrained(merged_dir)
			print(f"Saved merged model with LoRA weights at: {merged_dir}")
			print(f"Saved LoRA adapters at: {adapter_dir}")
		else:
			print(f"Saved LoRA adapters at: {adapter_dir}")
	else:
		print(f"Saved final model checkpoint at: {output_dir / 'final'}")

if __name__ == "__main__":
	main()
