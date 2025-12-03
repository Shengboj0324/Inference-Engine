"""Industrial-grade Style Transfer with LoRA (Low-Rank Adaptation).

Implements:
- LoRA for efficient fine-tuning of large language models
- Style-specific adapters for personalized content generation
- Parameter-efficient transfer learning
- Multi-style support with adapter switching
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StyleConfig(BaseModel):
    """Style configuration."""

    style_name: str
    description: str
    example_texts: List[str]
    parameters: Dict[str, Any] = {}


class GenerationResult(BaseModel):
    """Generation result with style."""

    text: str
    style: str
    confidence: float
    metadata: Dict[str, Any] = {}


@dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    r: int = 8  # Rank of low-rank matrices
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Modules to apply LoRA
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: str = "CAUSAL_LM"  # Task type
    inference_mode: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply to query and value projections
            self.target_modules = ["q_proj", "v_proj"]


class LoRAStyleTransfer:
    """LoRA-based style transfer for text generation.

    Features:
    - Parameter-efficient fine-tuning (< 1% of model parameters)
    - Multiple style adapters
    - Fast adapter switching
    - Preserves base model knowledge
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        config: Optional[LoRAConfig] = None,
    ):
        """Initialize LoRA style transfer.

        Args:
            base_model_name: Base model name
            config: LoRA configuration
        """
        self.base_model_name = base_model_name
        self.config = config or LoRAConfig()
        self.base_model = None
        self.tokenizer = None
        self.peft_model = None
        self.active_style = None
        self.style_adapters = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize base model and tokenizer."""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading base model: {self.base_model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype="auto",
                device_map="auto",
            )

            self._initialized = True
            logger.info("Base model initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Install with: pip install transformers peft torch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize base model: {e}")
            raise

    def create_style_adapter(
        self,
        style_config: StyleConfig,
        training_texts: List[str],
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        batch_size: int = 4,
    ) -> None:
        """Create and train LoRA adapter for specific style.

        Args:
            style_config: Style configuration
            training_texts: Training texts in target style
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        if not self._initialized:
            self.initialize()

        try:
            from peft import LoraConfig, get_peft_model, TaskType
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            from datasets import Dataset
            import torch

            logger.info(f"Creating LoRA adapter for style: {style_config.style_name}")

            # Create LoRA config
            peft_config = LoraConfig(
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias=self.config.bias,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
            )

            # Create PEFT model
            model = get_peft_model(self.base_model, peft_config)

            # Prepare dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                )

            dataset = Dataset.from_dict({"text": training_texts})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./lora_adapters/{style_config.style_name}",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                logging_steps=10,
                save_strategy="epoch",
                remove_unused_columns=False,
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            # Train
            logger.info(f"Training LoRA adapter for {num_epochs} epochs...")
            trainer.train()

            # Save adapter
            adapter_path = f"./lora_adapters/{style_config.style_name}"
            model.save_pretrained(adapter_path)

            self.style_adapters[style_config.style_name] = adapter_path

            logger.info(f"LoRA adapter saved to {adapter_path}")

        except ImportError as e:
            logger.error(f"Failed to import peft or datasets: {e}")
            logger.error("Install with: pip install peft datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to create style adapter: {e}")
            raise

    def load_style_adapter(self, style_name: str, adapter_path: str) -> None:
        """Load pre-trained style adapter.

        Args:
            style_name: Style name
            adapter_path: Path to adapter
        """
        if not self._initialized:
            self.initialize()

        try:
            logger.info(f"Loading style adapter: {style_name} from {adapter_path}")

            self.style_adapters[style_name] = adapter_path

            logger.info(f"Style adapter loaded: {style_name}")

        except Exception as e:
            logger.error(f"Failed to load style adapter: {e}")
            raise

    def switch_style(self, style_name: str) -> None:
        """Switch to different style adapter.

        Args:
            style_name: Style name to switch to
        """
        if style_name not in self.style_adapters:
            raise ValueError(f"Style {style_name} not found. Available: {list(self.style_adapters.keys())}")

        try:
            from peft import PeftModel

            logger.info(f"Switching to style: {style_name}")

            adapter_path = self.style_adapters[style_name]

            # Load PEFT model with adapter
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
            )

            self.active_style = style_name

            logger.info(f"Switched to style: {style_name}")

        except Exception as e:
            logger.error(f"Failed to switch style: {e}")
            raise

    def generate(
        self,
        prompt: str,
        style: Optional[str] = None,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[GenerationResult]:
        """Generate text in specified style.

        Args:
            prompt: Input prompt
            style: Style to use (if None, use active style)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generation results
        """
        if not self._initialized:
            self.initialize()

        # Switch style if needed
        if style and style != self.active_style:
            self.switch_style(style)

        if self.peft_model is None:
            raise ValueError("No style adapter loaded. Call switch_style() first.")

        try:
            import torch

            logger.info(f"Generating text in style: {self.active_style}")

            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move to device
            device = next(self.peft_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode
            results = []
            for i, output in enumerate(outputs):
                text = self.tokenizer.decode(output, skip_special_tokens=True)

                # Remove prompt from output
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()

                results.append(GenerationResult(
                    text=text,
                    style=self.active_style,
                    confidence=1.0 / (i + 1),  # Simple confidence based on rank
                    metadata={
                        'prompt': prompt,
                        'temperature': temperature,
                        'top_p': top_p,
                    },
                ))

            return results

        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise

    def transfer_style(
        self,
        source_text: str,
        target_style: str,
        max_length: int = 200,
    ) -> GenerationResult:
        """Transfer text to target style.

        Args:
            source_text: Source text
            target_style: Target style
            max_length: Maximum generation length

        Returns:
            Generation result in target style
        """
        # Create prompt for style transfer
        prompt = f"Rewrite the following text in {target_style} style:\n\n{source_text}\n\nRewritten:"

        # Generate
        results = self.generate(
            prompt=prompt,
            style=target_style,
            max_length=max_length,
            num_return_sequences=1,
        )

        return results[0] if results else None

    def get_available_styles(self) -> List[str]:
        """Get list of available styles.

        Returns:
            List of style names
        """
        return list(self.style_adapters.keys())

    def get_adapter_info(self, style_name: str) -> Dict[str, Any]:
        """Get information about style adapter.

        Args:
            style_name: Style name

        Returns:
            Dictionary of adapter information
        """
        if style_name not in self.style_adapters:
            raise ValueError(f"Style {style_name} not found")

        adapter_path = self.style_adapters[style_name]

        # Count parameters
        try:
            from peft import PeftModel

            peft_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
            )

            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in peft_model.parameters())

            return {
                'style_name': style_name,
                'adapter_path': adapter_path,
                'trainable_params': trainable_params,
                'total_params': total_params,
                'trainable_ratio': trainable_params / total_params,
                'lora_rank': self.config.r,
                'lora_alpha': self.config.lora_alpha,
            }

        except Exception as e:
            logger.error(f"Failed to get adapter info: {e}")
            return {
                'style_name': style_name,
                'adapter_path': adapter_path,
                'error': str(e),
            }

