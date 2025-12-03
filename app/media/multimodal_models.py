"""Industrial-grade multimodal models for image-text understanding.

Implements:
- CLIP: Contrastive Language-Image Pre-training for image-text alignment
- LLaVA: Large Language and Vision Assistant for visual question answering
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ImageTextAlignment(BaseModel):
    """Image-text alignment result from CLIP."""

    image_embedding: List[float]
    text_embeddings: Dict[str, List[float]]  # text -> embedding
    similarity_scores: Dict[str, float]  # text -> similarity score
    best_match: str
    best_score: float
    model_name: str = "openai/clip-vit-base-patch32"


class VisualQAResult(BaseModel):
    """Visual question answering result from LLaVA."""

    question: str
    answer: str
    confidence: float
    reasoning: Optional[str] = None
    model_name: str = "llava-1.5-7b"


@dataclass
class CLIPConfig:
    """Configuration for CLIP model."""

    model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    device: str = "cpu"  # "cuda" for GPU
    batch_size: int = 32


@dataclass
class LLaVAConfig:
    """Configuration for LLaVA model."""

    model_name: str = "llava-hf/llava-1.5-7b-hf"
    max_new_tokens: int = 512
    temperature: float = 0.2
    device: str = "cpu"  # "cuda" for GPU


class CLIPModel:
    """CLIP model for image-text alignment and zero-shot classification.

    Features:
    - Image-text similarity scoring
    - Zero-shot image classification
    - Cross-modal retrieval
    - Semantic image search
    - Content moderation
    """

    def __init__(self, config: Optional[CLIPConfig] = None):
        """Initialize CLIP model.

        Args:
            config: CLIP configuration
        """
        self.config = config or CLIPConfig()
        self.model = None
        self.processor = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize CLIP model and processor."""
        if self._initialized:
            return

        try:
            from transformers import CLIPModel as HFCLIPModel, CLIPProcessor

            logger.info(f"Loading CLIP model: {self.config.model_name}")

            # Load processor and model
            self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self.model = HFCLIPModel.from_pretrained(self.config.model_name)

            # Move to device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    logger.info("CLIP model loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("CLIP model loaded on CPU")

            self.model.eval()
            self._initialized = True

            logger.info("CLIP model initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Install with: pip install transformers torch pillow")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise

    async def align_image_text(
        self,
        image_path: str,
        texts: List[str],
    ) -> ImageTextAlignment:
        """Compute image-text alignment scores.

        Args:
            image_path: Path to image file
            texts: List of text descriptions to compare

        Returns:
            ImageTextAlignment with similarity scores
        """
        await self.initialize()

        try:
            import torch

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process inputs
            inputs = self.processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get embeddings
            image_embeds = outputs.image_embeds.cpu().numpy()[0]
            text_embeds = outputs.text_embeds.cpu().numpy()

            # Compute similarities (cosine similarity)
            # Normalize embeddings
            image_embeds_norm = image_embeds / np.linalg.norm(image_embeds)
            text_embeds_norm = text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)

            # Compute cosine similarity
            similarities = np.dot(text_embeds_norm, image_embeds_norm)

            # Build results
            text_embeddings = {text: emb.tolist() for text, emb in zip(texts, text_embeds)}
            similarity_scores = {text: float(sim) for text, sim in zip(texts, similarities)}

            # Find best match
            best_idx = np.argmax(similarities)
            best_match = texts[best_idx]
            best_score = float(similarities[best_idx])

            return ImageTextAlignment(
                image_embedding=image_embeds.tolist(),
                text_embeddings=text_embeddings,
                similarity_scores=similarity_scores,
                best_match=best_match,
                best_score=best_score,
                model_name=self.config.model_name,
            )

        except Exception as e:
            logger.error(f"Failed to align image-text: {e}")
            raise

    async def zero_shot_classify(
        self,
        image_path: str,
        candidate_labels: List[str],
        hypothesis_template: str = "a photo of {}",
    ) -> Dict[str, float]:
        """Perform zero-shot image classification.

        Args:
            image_path: Path to image file
            candidate_labels: List of candidate class labels
            hypothesis_template: Template for text prompts

        Returns:
            Dictionary mapping labels to probabilities
        """
        # Create text prompts from labels
        texts = [hypothesis_template.format(label) for label in candidate_labels]

        # Get alignment scores
        alignment = await self.align_image_text(image_path, texts)

        # Convert similarities to probabilities using softmax
        import numpy as np
        scores = np.array([alignment.similarity_scores[text] for text in texts])
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probs = exp_scores / exp_scores.sum()

        # Map back to labels
        return {label: float(prob) for label, prob in zip(candidate_labels, probs)}

    async def semantic_search(
        self,
        query_text: str,
        image_paths: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Search images by text query.

        Args:
            query_text: Text query
            image_paths: List of image paths to search
            top_k: Number of top results to return

        Returns:
            List of (image_path, score) tuples sorted by relevance
        """
        results = []

        for image_path in image_paths:
            try:
                alignment = await self.align_image_text(image_path, [query_text])
                score = alignment.similarity_scores[query_text]
                results.append((image_path, score))
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


class LLaVAModel:
    """LLaVA model for visual question answering and image understanding.

    Features:
    - Visual question answering
    - Image captioning
    - Visual reasoning
    - Detailed image description
    - Multi-turn visual dialogue
    """

    def __init__(self, config: Optional[LLaVAConfig] = None):
        """Initialize LLaVA model.

        Args:
            config: LLaVA configuration
        """
        self.config = config or LLaVAConfig()
        self.model = None
        self.processor = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize LLaVA model and processor."""
        if self._initialized:
            return

        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor

            logger.info(f"Loading LLaVA model: {self.config.model_name}")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )

            # Move to device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    logger.info("LLaVA model loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("LLaVA model loaded on CPU")

            self.model.eval()
            self._initialized = True

            logger.info("LLaVA model initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Install with: pip install transformers torch pillow accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLaVA model: {e}")
            raise

    async def answer_question(
        self,
        image_path: str,
        question: str,
    ) -> VisualQAResult:
        """Answer a question about an image.

        Args:
            image_path: Path to image file
            question: Question about the image

        Returns:
            VisualQAResult with answer and confidence
        """
        await self.initialize()

        try:
            import torch

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Create prompt
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

            # Move to device
            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True if self.config.temperature > 0 else False,
                )

            # Decode answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Extract answer (remove prompt)
            if "ASSISTANT:" in answer:
                answer = answer.split("ASSISTANT:")[-1].strip()

            # Estimate confidence (placeholder - would need proper calibration)
            confidence = 0.8 if len(answer) > 10 else 0.5

            return VisualQAResult(
                question=question,
                answer=answer,
                confidence=confidence,
                model_name=self.config.model_name,
            )

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise

    async def generate_caption(
        self,
        image_path: str,
        detailed: bool = False,
    ) -> str:
        """Generate caption for an image.

        Args:
            image_path: Path to image file
            detailed: Whether to generate detailed description

        Returns:
            Generated caption
        """
        if detailed:
            question = "Describe this image in detail, including all visible objects, people, actions, and context."
        else:
            question = "Describe this image briefly."

        result = await self.answer_question(image_path, question)
        return result.answer

