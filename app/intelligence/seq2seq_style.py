"""Industrial-grade Sequence-to-Sequence with Style Control.

Implements:
- Transformer-based seq2seq models
- Controllable text generation with style attributes
- Multi-task learning for different generation tasks
- Beam search and nucleus sampling
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StyleAttributes(BaseModel):
    """Style control attributes."""

    formality: float = 0.5  # 0=casual, 1=formal
    sentiment: float = 0.5  # 0=negative, 1=positive
    complexity: float = 0.5  # 0=simple, 1=complex
    length: str = "medium"  # "short", "medium", "long"
    tone: str = "neutral"  # "neutral", "professional", "friendly", "humorous"


class GenerationConfig(BaseModel):
    """Generation configuration."""

    max_length: int = 512
    min_length: int = 10
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3


class Seq2SeqResult(BaseModel):
    """Seq2seq generation result."""

    generated_text: str
    style_attributes: StyleAttributes
    confidence: float
    metadata: Dict[str, Any] = {}


@dataclass
class Seq2SeqConfig:
    """Configuration for seq2seq model."""

    model_name: str = "facebook/bart-large-cnn"  # Base model
    max_source_length: int = 1024
    max_target_length: int = 512
    device: str = "cpu"
    use_style_control: bool = True
    num_style_tokens: int = 5  # Number of special tokens for style control


class ControlledSeq2Seq:
    def __init__(self, config: Optional[Seq2SeqConfig] = None):
        """Initialize controlled seq2seq model.

        Args:
            config: Seq2seq configuration
        """
        self.config = config or Seq2SeqConfig()
        self.model = None
        self.tokenizer = None
        self.style_embeddings = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize model and tokenizer."""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            import torch.nn as nn

            logger.info(f"Loading seq2seq model: {self.config.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",
            )

            # Add style control tokens if enabled
            if self.config.use_style_control:
                style_tokens = [
                    "<FORMAL>", "<CASUAL>",
                    "<POSITIVE>", "<NEGATIVE>", "<NEUTRAL>",
                    "<SIMPLE>", "<COMPLEX>",
                    "<SHORT>", "<MEDIUM>", "<LONG>",
                    "<PROFESSIONAL>", "<FRIENDLY>", "<HUMOROUS>",
                ]

                # Add special tokens
                num_added = self.tokenizer.add_special_tokens({
                    'additional_special_tokens': style_tokens
                })

                if num_added > 0:
                    # Resize model embeddings
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(f"Added {num_added} style control tokens")

            # Move to device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    logger.info("Model loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("Model loaded on CPU")

            self._initialized = True
            logger.info("Seq2seq model initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _prepare_style_prefix(self, style: StyleAttributes) -> str:
        """Prepare style control prefix.

        Args:
            style: Style attributes

        Returns:
            Style prefix string
        """
        if not self.config.use_style_control:
            return ""

        tokens = []

        # Formality
        if style.formality > 0.7:
            tokens.append("<FORMAL>")
        elif style.formality < 0.3:
            tokens.append("<CASUAL>")

        # Sentiment
        if style.sentiment > 0.7:
            tokens.append("<POSITIVE>")
        elif style.sentiment < 0.3:
            tokens.append("<NEGATIVE>")
        else:
            tokens.append("<NEUTRAL>")

        # Complexity
        if style.complexity > 0.7:
            tokens.append("<COMPLEX>")
        elif style.complexity < 0.3:
            tokens.append("<SIMPLE>")

        # Length
        if style.length == "short":
            tokens.append("<SHORT>")
        elif style.length == "long":
            tokens.append("<LONG>")
        else:
            tokens.append("<MEDIUM>")

        # Tone
        if style.tone == "professional":
            tokens.append("<PROFESSIONAL>")
        elif style.tone == "friendly":
            tokens.append("<FRIENDLY>")
        elif style.tone == "humorous":
            tokens.append("<HUMOROUS>")

        return " ".join(tokens) + " "

    def generate(
        self,
        source_text: str,
        style: Optional[StyleAttributes] = None,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Seq2SeqResult:
        """Generate text with style control.

        Args:
            source_text: Source text
            style: Style attributes
            gen_config: Generation configuration

        Returns:
            Generation result
        """
        if not self._initialized:
            self.initialize()

        style = style or StyleAttributes()
        gen_config = gen_config or GenerationConfig()

        try:
            import torch

            logger.info("Generating text with style control")

            # Prepare input with style prefix
            style_prefix = self._prepare_style_prefix(style)
            input_text = style_prefix + source_text

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.config.max_source_length,
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=gen_config.max_length,
                    min_length=gen_config.min_length,
                    num_beams=gen_config.num_beams,
                    temperature=gen_config.temperature,
                    top_k=gen_config.top_k,
                    top_p=gen_config.top_p,
                    repetition_penalty=gen_config.repetition_penalty,
                    length_penalty=gen_config.length_penalty,
                    no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
                    early_stopping=True,
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Compute confidence (simplified)
            confidence = 0.8  # Placeholder

            return Seq2SeqResult(
                generated_text=generated_text,
                style_attributes=style,
                confidence=confidence,
                metadata={
                    'source_length': len(source_text),
                    'generated_length': len(generated_text),
                    'num_beams': gen_config.num_beams,
                },
            )

        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise

    def summarize(
        self,
        text: str,
        style: Optional[StyleAttributes] = None,
        max_length: int = 150,
    ) -> Seq2SeqResult:
        """Generate summary with style control.

        Args:
            text: Text to summarize
            style: Style attributes
            max_length: Maximum summary length

        Returns:
            Summary result
        """
        gen_config = GenerationConfig(
            max_length=max_length,
            min_length=max_length // 4,
            num_beams=4,
            length_penalty=2.0,
        )

        return self.generate(text, style, gen_config)

    def paraphrase(
        self,
        text: str,
        style: Optional[StyleAttributes] = None,
    ) -> Seq2SeqResult:
        """Paraphrase text with style control.

        Args:
            text: Text to paraphrase
            style: Style attributes

        Returns:
            Paraphrase result
        """
        # Add paraphrase instruction
        source_text = f"Paraphrase: {text}"

        gen_config = GenerationConfig(
            max_length=len(text.split()) * 2,
            num_beams=5,
            temperature=0.8,
        )

        return self.generate(source_text, style, gen_config)

    def translate_style(
        self,
        text: str,
        source_style: StyleAttributes,
        target_style: StyleAttributes,
    ) -> Seq2SeqResult:
        """Translate text from one style to another.

        Args:
            text: Source text
            source_style: Source style
            target_style: Target style

        Returns:
            Translated result
        """
        # Use target style for generation
        return self.generate(text, target_style)

    def batch_generate(
        self,
        texts: List[str],
        styles: Optional[List[StyleAttributes]] = None,
        gen_config: Optional[GenerationConfig] = None,
    ) -> List[Seq2SeqResult]:
        """Generate text for multiple inputs.

        Args:
            texts: List of source texts
            styles: List of style attributes (one per text)
            gen_config: Generation configuration

        Returns:
            List of generation results
        """
        if not self._initialized:
            self.initialize()

        if styles is None:
            styles = [StyleAttributes()] * len(texts)

        gen_config = gen_config or GenerationConfig()

        try:
            import torch

            logger.info(f"Batch generating {len(texts)} texts")

            # Prepare inputs with style prefixes
            input_texts = []
            for text, style in zip(texts, styles):
                style_prefix = self._prepare_style_prefix(style)
                input_texts.append(style_prefix + text)

            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                max_length=self.config.max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=gen_config.max_length,
                    min_length=gen_config.min_length,
                    num_beams=gen_config.num_beams,
                    temperature=gen_config.temperature,
                    top_k=gen_config.top_k,
                    top_p=gen_config.top_p,
                    repetition_penalty=gen_config.repetition_penalty,
                    length_penalty=gen_config.length_penalty,
                    no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
                    early_stopping=True,
                )

            # Decode all outputs
            results = []
            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)

                results.append(Seq2SeqResult(
                    generated_text=generated_text,
                    style_attributes=styles[i],
                    confidence=0.8,
                    metadata={
                        'source_length': len(texts[i]),
                        'generated_length': len(generated_text),
                    },
                ))

            return results

        except Exception as e:
            logger.error(f"Failed to batch generate: {e}")
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        if not self._initialized:
            raise ValueError("Model not initialized")

        try:
            from pathlib import Path

            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            logger.info(f"Saved model to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Loading model from {path}")

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)

            # Move to device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")

            self._initialized = True

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

