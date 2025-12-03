"""Industrial-grade Vision Transformer (ViT) for advanced image understanding.

Implements Vision Transformer architecture for superior image analysis beyond traditional CNNs.
Provides patch-based image encoding, attention-based feature extraction, and semantic understanding.
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


class ImageFeatures(BaseModel):
    """Extracted image features from ViT."""

    embedding: List[float]  # Global image embedding
    patch_embeddings: Optional[List[List[float]]] = None  # Per-patch embeddings
    attention_maps: Optional[List[List[float]]] = None  # Attention weights
    semantic_labels: List[str] = []  # Predicted labels
    confidence_scores: List[float] = []  # Confidence per label
    model_name: str = "vit-base-patch16-224"


class SceneUnderstanding(BaseModel):
    """High-level scene understanding from ViT."""

    scene_type: str  # indoor, outdoor, urban, nature, etc.
    scene_confidence: float
    objects_detected: List[str]
    spatial_relationships: List[str]  # "person left of car", etc.
    contextual_description: str


@dataclass
class ViTConfig:
    """Configuration for Vision Transformer."""

    model_name: str = "google/vit-base-patch16-224"
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 196  # (224/16)^2
    embedding_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    enable_attention_maps: bool = False
    enable_patch_embeddings: bool = False
    device: str = "cpu"  # "cuda" for GPU


class VisionTransformer:
    """Industrial-grade Vision Transformer for image understanding.

    Features:
    - Patch-based image encoding
    - Multi-head self-attention
    - Global and local feature extraction
    - Semantic label prediction
    - Scene understanding
    - Attention visualization
    """

    def __init__(self, config: Optional[ViTConfig] = None):
        """Initialize Vision Transformer.

        Args:
            config: ViT configuration
        """
        self.config = config or ViTConfig()
        self.model = None
        self.processor = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize ViT model and processor.

        Lazy initialization to avoid loading model on import.
        """
        if self._initialized:
            return

        try:
            from transformers import ViTForImageClassification, ViTImageProcessor

            logger.info(f"Loading ViT model: {self.config.model_name}")

            # Load processor and model
            self.processor = ViTImageProcessor.from_pretrained(self.config.model_name)
            self.model = ViTForImageClassification.from_pretrained(
                self.config.model_name,
                output_attentions=self.config.enable_attention_maps,
                output_hidden_states=True,
            )

            # Move to device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    logger.info("ViT model loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("ViT model loaded on CPU")

            self.model.eval()  # Set to evaluation mode
            self._initialized = True

            logger.info("ViT model initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Install with: pip install transformers torch pillow")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ViT model: {e}")
            raise

    async def extract_features(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> ImageFeatures:
        """Extract features from image using ViT.

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            ImageFeatures with embeddings and predictions
        """
        await self.initialize()

        try:
            import torch

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Process image
            inputs = self.processor(images=image, return_tensors="pt")

            # Move to device
            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract logits and hidden states
            logits = outputs.logits
            hidden_states = outputs.hidden_states  # Tuple of layer outputs

            # Get global embedding (CLS token from last layer)
            # Shape: [batch_size, num_patches + 1, embedding_dim]
            last_hidden_state = hidden_states[-1]
            cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()[0]  # CLS token

            # Get top-k predictions
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)

            # Convert to labels
            semantic_labels = []
            confidence_scores = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                label = self.model.config.id2label[idx.item()]
                semantic_labels.append(label)
                confidence_scores.append(float(prob.item()))

            # Extract patch embeddings if enabled
            patch_embeddings = None
            if self.config.enable_patch_embeddings:
                # Get patch tokens (exclude CLS token)
                patch_tokens = last_hidden_state[:, 1:, :].cpu().numpy()[0]
                patch_embeddings = patch_tokens.tolist()

            # Extract attention maps if enabled
            attention_maps = None
            if self.config.enable_attention_maps and outputs.attentions:
                # Get attention from last layer, average across heads
                last_attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
                avg_attention = last_attention.mean(dim=1).cpu().numpy()[0]  # [seq, seq]
                attention_maps = avg_attention.tolist()

            return ImageFeatures(
                embedding=cls_embedding.tolist(),
                patch_embeddings=patch_embeddings,
                attention_maps=attention_maps,
                semantic_labels=semantic_labels,
                confidence_scores=confidence_scores,
                model_name=self.config.model_name,
            )

        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
            raise

    async def understand_scene(
        self,
        image_path: str,
    ) -> SceneUnderstanding:
        """Perform high-level scene understanding.

        Args:
            image_path: Path to image file

        Returns:
            SceneUnderstanding with scene type, objects, relationships
        """
        # Extract features first
        features = await self.extract_features(image_path, top_k=10)

        # Analyze semantic labels to determine scene type
        scene_type = self._infer_scene_type(features.semantic_labels)
        scene_confidence = features.confidence_scores[0] if features.confidence_scores else 0.0

        # Extract objects from labels
        objects_detected = [
            label for label in features.semantic_labels
            if not self._is_scene_label(label)
        ]

        # Generate contextual description
        contextual_description = self._generate_context_description(
            scene_type,
            objects_detected,
            features.confidence_scores[:len(objects_detected)]
        )

        return SceneUnderstanding(
            scene_type=scene_type,
            scene_confidence=scene_confidence,
            objects_detected=objects_detected[:5],  # Top 5 objects
            spatial_relationships=[],  # Would need object detection for this
            contextual_description=contextual_description,
        )

    def _infer_scene_type(self, labels: List[str]) -> str:
        """Infer scene type from semantic labels."""
        # Scene type keywords
        indoor_keywords = ["room", "kitchen", "bedroom", "office", "indoor"]
        outdoor_keywords = ["outdoor", "landscape", "sky", "mountain", "beach"]
        urban_keywords = ["street", "building", "city", "urban", "road"]
        nature_keywords = ["forest", "tree", "nature", "wildlife", "plant"]

        # Count matches
        label_text = " ".join(labels).lower()

        if any(kw in label_text for kw in indoor_keywords):
            return "indoor"
        elif any(kw in label_text for kw in urban_keywords):
            return "urban"
        elif any(kw in label_text for kw in nature_keywords):
            return "nature"
        elif any(kw in label_text for kw in outdoor_keywords):
            return "outdoor"
        else:
            return "general"

    def _is_scene_label(self, label: str) -> bool:
        """Check if label represents a scene rather than an object."""
        scene_keywords = [
            "scene", "landscape", "indoor", "outdoor", "room",
            "background", "setting", "environment"
        ]
        return any(kw in label.lower() for kw in scene_keywords)

    def _generate_context_description(
        self,
        scene_type: str,
        objects: List[str],
        confidences: List[float],
    ) -> str:
        """Generate contextual description from scene and objects."""
        if not objects:
            return f"A {scene_type} scene"

        # Get high-confidence objects
        high_conf_objects = [
            obj for obj, conf in zip(objects, confidences)
            if conf > 0.1
        ]

        if not high_conf_objects:
            high_conf_objects = objects[:2]

        if len(high_conf_objects) == 1:
            return f"A {scene_type} scene featuring {high_conf_objects[0]}"
        elif len(high_conf_objects) == 2:
            return f"A {scene_type} scene with {high_conf_objects[0]} and {high_conf_objects[1]}"
        else:
            obj_list = ", ".join(high_conf_objects[:-1])
            return f"A {scene_type} scene containing {obj_list}, and {high_conf_objects[-1]}"

    async def batch_extract_features(
        self,
        image_paths: List[str],
        max_concurrent: int = 4,
    ) -> List[ImageFeatures]:
        """Extract features from multiple images concurrently.

        Args:
            image_paths: List of image file paths
            max_concurrent: Maximum concurrent processing tasks

        Returns:
            List of ImageFeatures
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _extract_with_semaphore(path: str) -> Optional[ImageFeatures]:
            async with semaphore:
                try:
                    return await self.extract_features(path)
                except Exception as e:
                    logger.error(f"Failed to extract features from {path}: {e}")
                    return None

        tasks = [_extract_with_semaphore(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        return [r for r in results if isinstance(r, ImageFeatures)]

    def get_statistics(self) -> Dict[str, Any]:
        """Get ViT model statistics."""
        return {
            "model_name": self.config.model_name,
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "num_patches": self.config.num_patches,
            "embedding_dim": self.config.embedding_dim,
            "num_heads": self.config.num_heads,
            "num_layers": self.config.num_layers,
            "device": self.config.device,
            "initialized": self._initialized,
        }

