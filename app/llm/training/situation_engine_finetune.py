"""SituationEngine-specific QLoRA fine-tuning orchestrator.

This module is a thin adapter on top of ``LoRATrainer``. It:

1. Defines the QLoRA defaults used for Llama 3.1 8B on a single A100.
2. Runs a CPU-only pre-flight check over the JSONL produced by
   ``scripts/build_training_set.py`` so misformatted data is caught
   before a single GPU minute is spent.
3. Provides an assistant-only tokenization function that masks the
   prompt and system turns out of the loss, so the trainee only learns
   to generate the assistant JSON.

Heavy imports (``transformers``, ``peft``, ``torch``, ``datasets``) are
deferred inside the methods that need them so the pre-flight stage can
run inside CI without GPU wheels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.intelligence.situation_report import SituationReport
from app.llm.training.situation_prompt import (
    MissingSystemPromptError,
    load_system_prompt,
)

logger = logging.getLogger(__name__)


# A100-friendly QLoRA defaults for Llama 3.1 8B Instruct. These are
# starting points; the Colab notebook overrides via CLI when sweeping.
_DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@dataclass
class SituationEngineFineTuneConfig:
    """Configuration consumed by :class:`SituationEngineFineTuner`."""

    base_model: str = _DEFAULT_BASE_MODEL
    train_file: Path = Path("data/training/train.jsonl")
    val_file: Optional[Path] = Path("data/training/val.jsonl")
    output_dir: Path = Path("checkpoints/situation_engine")
    system_prompt_path: Optional[Path] = None

    # Sequence length budget: observations + report fit in 4k tokens for
    # the vast majority of scenarios; we cap to keep memory predictable.
    max_seq_length: int = 4096

    # QLoRA defaults for A100 40GB.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.5e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 5

    # Hard pre-flight knob: at least this many usable rows must be in the
    # training file or the run is aborted (catches an empty train.jsonl).
    min_train_examples: int = 1


@dataclass(frozen=True)
class PreflightReport:
    """Result of the CPU-only data-and-prompt sanity pass."""

    train_examples: int
    val_examples: int
    system_prompt_chars: int
    assistant_min_chars: int
    assistant_max_chars: int


class SituationEngineFineTuner:
    """Orchestrates the SituationEngine QLoRA run with safe defaults.

    Typical usage on Colab::

        cfg = SituationEngineFineTuneConfig(
            train_file=Path("data/training/train.jsonl"),
            val_file=Path("data/training/val.jsonl"),
            output_dir=Path("/content/drive/MyDrive/se_ckpts"),
        )
        ft = SituationEngineFineTuner(cfg)
        ft.preflight()                # CPU-only, no GPU required
        ft.run()                      # loads model, trains, saves
    """

    def __init__(self, config: SituationEngineFineTuneConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # CPU-only pre-flight
    # ------------------------------------------------------------------ #

    def preflight(self) -> PreflightReport:
        """Validate prompt + training files without loading the model."""
        try:
            system_prompt = load_system_prompt(self.config.system_prompt_path)
        except MissingSystemPromptError:
            raise

        train_examples = self._validate_jsonl(
            self.config.train_file,
            kind="train",
            min_required=self.config.min_train_examples,
            system_prompt=system_prompt,
        )
        val_examples = 0
        if self.config.val_file is not None and self.config.val_file.exists():
            val_examples = self._validate_jsonl(
                self.config.val_file,
                kind="val",
                min_required=0,
                system_prompt=system_prompt,
            )

        assistant_lens = self._assistant_lengths(self.config.train_file)
        report = PreflightReport(
            train_examples=train_examples,
            val_examples=val_examples,
            system_prompt_chars=len(system_prompt),
            assistant_min_chars=min(assistant_lens) if assistant_lens else 0,
            assistant_max_chars=max(assistant_lens) if assistant_lens else 0,
        )
        logger.info("preflight ok: %s", report)
        return report

    def _validate_jsonl(
        self,
        path: Path,
        *,
        kind: str,
        min_required: int,
        system_prompt: str,
    ) -> int:
        if not path.exists():
            raise FileNotFoundError(f"{kind} file does not exist: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                count += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{lineno} not valid JSON: {exc}"
                    ) from exc
                self._validate_row(row, system_prompt, path, lineno)
        if count < min_required:
            raise ValueError(
                f"{path}: contains {count} examples, need at least "
                f"{min_required}"
            )
        return count

    def _validate_row(
        self,
        row: Dict[str, Any],
        system_prompt: str,
        path: Path,
        lineno: int,
    ) -> None:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(
                f"{path}:{lineno} messages must be a list of 3 entries "
                "(system, user, assistant)"
            )
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(
                f"{path}:{lineno} message roles must be "
                f"[system, user, assistant], got {roles}"
            )
        if messages[0]["content"].strip() != system_prompt.strip():
            raise ValueError(
                f"{path}:{lineno} system prompt does not match the frozen "
                "prompt; rebuild training set with the current prompt"
            )
        try:
            user_payload = json.loads(messages[1]["content"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{lineno} user message is not valid JSON: {exc}"
            ) from exc
        if "observations" not in user_payload:
            raise ValueError(
                f"{path}:{lineno} user message JSON missing 'observations'"
            )
        try:
            assistant_payload = json.loads(messages[2]["content"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{lineno} assistant message is not valid JSON: {exc}"
            ) from exc
        SituationReport.model_validate(assistant_payload)

    def _assistant_lengths(self, path: Path) -> List[int]:
        lengths: List[int] = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                lengths.append(len(row["messages"][2]["content"]))
        return lengths

    # ------------------------------------------------------------------ #
    # GPU run (deferred imports)
    # ------------------------------------------------------------------ #

    def build_trainer_config(self):
        """Construct the underlying ``LoRATrainingConfig`` from our defaults."""
        from app.llm.training.lora_trainer import LoRATrainingConfig

        return LoRATrainingConfig(
            base_model=self.config.base_model,
            model_max_length=self.config.max_seq_length,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.target_modules),
            use_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            use_nested_quant=True,
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            bf16=True,
            fp16=False,
        )

    def run(self) -> Dict[str, Any]:
        """Execute the full QLoRA fine-tune. Requires GPU + GPU wheels."""
        from app.llm.training.lora_trainer import LoRATrainer

        self.preflight()
        trainer_cfg = self.build_trainer_config()
        trainer = LoRATrainer(trainer_cfg)
        trainer.load_model()
        train_ds, val_ds = self._build_datasets(trainer.tokenizer)
        metrics = trainer.train(train_dataset=train_ds, val_dataset=val_ds)
        trainer.save_model(str(self.config.output_dir / "final"))
        self._write_run_manifest(metrics)
        return metrics

    def _build_datasets(self, tokenizer):
        """Tokenize train/val with assistant-only loss masking.

        Tokens belonging to the system prompt and the user observations
        are set to ``-100`` so the trainer ignores them in the loss; only
        the assistant JSON contributes gradient.
        """
        from datasets import load_dataset

        data_files = {"train": str(self.config.train_file)}
        if self.config.val_file and self.config.val_file.exists():
            data_files["validation"] = str(self.config.val_file)
        ds = load_dataset("json", data_files=data_files)

        max_len = self.config.max_seq_length

        def tokenize(batch):
            input_ids_b: List[List[int]] = []
            labels_b: List[List[int]] = []
            attention_b: List[List[int]] = []
            for messages in batch["messages"]:
                ids, labels = _encode_with_assistant_mask(
                    tokenizer, messages, max_len
                )
                attn = [1] * len(ids)
                pad = max_len - len(ids)
                if pad > 0:
                    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                    ids = ids + [pad_id] * pad
                    labels = labels + [-100] * pad
                    attn = attn + [0] * pad
                input_ids_b.append(ids)
                labels_b.append(labels)
                attention_b.append(attn)
            return {
                "input_ids": input_ids_b,
                "labels": labels_b,
                "attention_mask": attention_b,
            }

        tokenized = ds.map(
            tokenize, batched=True, remove_columns=ds["train"].column_names,
        )
        return tokenized["train"], tokenized.get("validation")

    def _write_run_manifest(self, metrics: Dict[str, Any]) -> None:
        manifest_path = self.config.output_dir / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_model": self.config.base_model,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "num_train_epochs": self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate,
            "metrics": {k: float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float))},
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote run manifest to %s", manifest_path)


def _encode_with_assistant_mask(
    tokenizer, messages: List[Dict[str, str]], max_len: int,
):
    """Tokenize a 3-message conversation, masking everything but assistant.

    Falls back to a robust prefix-vs-full diff strategy that works on any
    HuggingFace chat-template, including those that do not yet expose
    ``return_assistant_tokens_mask``.
    """
    user_only = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
    )
    if full[: len(user_only)] != user_only:
        # If the template inserts a different prefix for the full sequence
        # (rare), fall back to "label everything" rather than silently
        # mis-mask; an alarm in logs is better than a wrong loss.
        logger.warning(
            "assistant-mask prefix mismatch; falling back to full-sequence loss"
        )
        labels = list(full)
    else:
        labels = [-100] * len(user_only) + list(full[len(user_only):])

    ids = list(full)
    if len(ids) > max_len:
        ids = ids[:max_len]
        labels = labels[:max_len]
    return ids, labels


