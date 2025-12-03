"""Industrial-grade multimodal embeddings for unified text-image-video representation.

Implements unified embedding space where text, images, and videos can be compared
and searched semantically across modalities.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Type of content modality."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MultimodalEmbedding(BaseModel):
    """Unified multimodal embedding."""

    embedding: List[float]
    modality: ModalityType
    source_id: str
    metadata: Dict[str, Any] = {}
    model_name: str = "clip"


class CrossModalSearchResult(BaseModel):
    """Cross-modal search result."""

    source_id: str
    modality: ModalityType
    similarity: float
    metadata: Dict[str, Any] = {}


@dataclass
class MultimodalConfig:
    """Configuration for multimodal embeddings."""

    text_model: str = "openai/clip-vit-base-patch32"
    image_model: str = "openai/clip-vit-base-patch32"
    video_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    normalize_embeddings: bool = True
    device: str = "cpu"


class MultimodalEmbeddingEngine:
    """Unified embedding engine for text, images, and videos.

    Features:
    - Unified embedding space (CLIP-based)
    - Cross-modal search (text → image, image → text, etc.)
    - Semantic similarity across modalities
    - Efficient batch processing
    - Embedding fusion for multi-modal content
    """

    def __init__(self, config: Optional[MultimodalConfig] = None):
        """Initialize multimodal embedding engine.

        Args:
            config: Multimodal configuration
        """
        self.config = config or MultimodalConfig()
        self.clip_model = None
        self.clip_processor = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize embedding models."""
        if self._initialized:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading CLIP model: {self.config.text_model}")

            # Load CLIP (handles both text and images)
            self.clip_processor = CLIPProcessor.from_pretrained(self.config.text_model)
            self.clip_model = CLIPModel.from_pretrained(self.config.text_model)

            # Move to device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.clip_model = self.clip_model.to("cuda")
                    logger.info("CLIP model loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("CLIP model loaded on CPU")

            self.clip_model.eval()
            self._initialized = True

            logger.info("Multimodal embedding engine initialized")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.error("Install with: pip install transformers torch pillow")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embedding engine: {e}")
            raise

    async def embed_text(
        self,
        text: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalEmbedding:
        """Generate embedding for text.

        Args:
            text: Text to embed
            source_id: Unique identifier
            metadata: Optional metadata

        Returns:
            MultimodalEmbedding
        """
        await self.initialize()

        try:
            import torch

            # Process text
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)

            # Move to device
            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)

            # Normalize if configured
            if self.config.normalize_embeddings:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to list
            embedding = text_features.cpu().numpy()[0].tolist()

            return MultimodalEmbedding(
                embedding=embedding,
                modality=ModalityType.TEXT,
                source_id=source_id,
                metadata=metadata or {},
                model_name=self.config.text_model,
            )

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    async def embed_image(
        self,
        image_path: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalEmbedding:
        """Generate embedding for image.

        Args:
            image_path: Path to image file
            source_id: Unique identifier
            metadata: Optional metadata

        Returns:
            MultimodalEmbedding
        """
        await self.initialize()

        try:
            import torch
            from PIL import Image

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process image
            inputs = self.clip_processor(images=image, return_tensors="pt")

            # Move to device
            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            # Normalize if configured
            if self.config.normalize_embeddings:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to list
            embedding = image_features.cpu().numpy()[0].tolist()

            return MultimodalEmbedding(
                embedding=embedding,
                modality=ModalityType.IMAGE,
                source_id=source_id,
                metadata=metadata or {},
                model_name=self.config.image_model,
            )

        except Exception as e:
            logger.error(f"Failed to embed image: {e}")
            raise

    async def embed_video(
        self,
        video_path: str,
        source_id: str,
        num_frames: int = 8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalEmbedding:
        """Generate embedding for video by sampling frames.

        Args:
            video_path: Path to video file
            source_id: Unique identifier
            num_frames: Number of frames to sample
            metadata: Optional metadata

        Returns:
            MultimodalEmbedding (averaged from frame embeddings)
        """
        await self.initialize()

        try:
            import cv2
            import torch
            from PIL import Image

            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frame_embeddings = []

            for frame_idx in frame_indices:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # Process frame
                inputs = self.clip_processor(images=image, return_tensors="pt")

                # Move to device
                if self.config.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # Generate embedding
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)

                # Normalize
                if self.config.normalize_embeddings:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                frame_embeddings.append(image_features.cpu().numpy()[0])

            cap.release()

            if not frame_embeddings:
                raise ValueError("No frames could be extracted from video")

            # Average frame embeddings
            avg_embedding = np.mean(frame_embeddings, axis=0)

            # Normalize averaged embedding
            if self.config.normalize_embeddings:
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            return MultimodalEmbedding(
                embedding=avg_embedding.tolist(),
                modality=ModalityType.VIDEO,
                source_id=source_id,
                metadata=metadata or {},
                model_name=self.config.video_model,
            )

        except Exception as e:
            logger.error(f"Failed to embed video: {e}")
            raise

    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (-1 to 1)
        """
        # Convert to numpy
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    async def cross_modal_search(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[MultimodalEmbedding],
        top_k: int = 10,
        filter_modality: Optional[ModalityType] = None,
    ) -> List[CrossModalSearchResult]:
        """Search across modalities.

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of results to return
            filter_modality: Optional modality filter

        Returns:
            List of CrossModalSearchResult sorted by similarity
        """
        results = []

        for candidate in candidate_embeddings:
            # Apply modality filter
            if filter_modality and candidate.modality != filter_modality:
                continue

            # Compute similarity
            similarity = await self.compute_similarity(
                query_embedding,
                candidate.embedding,
            )

            results.append(CrossModalSearchResult(
                source_id=candidate.source_id,
                modality=candidate.modality,
                similarity=similarity,
                metadata=candidate.metadata,
            ))

        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:top_k]

    async def fuse_embeddings(
        self,
        embeddings: List[List[float]],
        weights: Optional[List[float]] = None,
    ) -> List[float]:
        """Fuse multiple embeddings into one.

        Args:
            embeddings: List of embeddings to fuse
            weights: Optional weights for each embedding

        Returns:
            Fused embedding
        """
        if not embeddings:
            raise ValueError("No embeddings to fuse")

        # Convert to numpy
        emb_array = np.array(embeddings)

        # Apply weights if provided
        if weights:
            if len(weights) != len(embeddings):
                raise ValueError("Number of weights must match number of embeddings")
            weights_array = np.array(weights).reshape(-1, 1)
            emb_array = emb_array * weights_array

        # Average
        fused = np.mean(emb_array, axis=0)

        # Normalize
        if self.config.normalize_embeddings:
            fused = fused / np.linalg.norm(fused)

        return fused.tolist()

    async def batch_embed_texts(
        self,
        texts: List[str],
        source_ids: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[MultimodalEmbedding]:
        """Batch embed multiple texts.

        Args:
            texts: List of texts
            source_ids: List of source IDs
            metadata_list: Optional list of metadata dicts

        Returns:
            List of MultimodalEmbedding
        """
        await self.initialize()

        try:
            import torch

            # Process texts
            inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)

            # Move to device
            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)

            # Normalize if configured
            if self.config.normalize_embeddings:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to list of embeddings
            embeddings_np = text_features.cpu().numpy()

            results = []
            for i, (text, source_id) in enumerate(zip(texts, source_ids)):
                metadata = metadata_list[i] if metadata_list else {}
                results.append(MultimodalEmbedding(
                    embedding=embeddings_np[i].tolist(),
                    modality=ModalityType.TEXT,
                    source_id=source_id,
                    metadata=metadata,
                    model_name=self.config.text_model,
                ))

            return results

        except Exception as e:
            logger.error(f"Failed to batch embed texts: {e}")
            raise

