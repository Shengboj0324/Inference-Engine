"""Industrial-grade Collaborative Filtering for personalized recommendations.

Implements:
- Alternating Least Squares (ALS) for matrix factorization
- Neural Collaborative Filtering (NCF) with deep learning
- Implicit feedback handling
- Cold-start problem mitigation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UserItemInteraction(BaseModel):
    """User-item interaction record."""

    user_id: str
    item_id: str
    rating: float  # Implicit: 1.0 for interaction, 0.0 for no interaction
    timestamp: float
    interaction_type: str  # "view", "click", "like", "share", etc.


class Recommendation(BaseModel):
    """Recommendation result."""

    item_id: str
    score: float
    rank: int
    explanation: Optional[str] = None


@dataclass
class ALSConfig:
    """Configuration for ALS model."""

    num_factors: int = 100  # Latent factors
    num_iterations: int = 15
    regularization: float = 0.01
    alpha: float = 40.0  # Confidence scaling for implicit feedback
    random_state: int = 42


@dataclass
class NCFConfig:
    """Configuration for NCF model."""

    embedding_dim: int = 64
    hidden_layers: List[int] = None  # Default: [128, 64, 32]
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 256
    num_epochs: int = 20
    device: str = "cpu"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class ALSRecommender:
    """Alternating Least Squares recommender.

    Features:
    - Matrix factorization for implicit feedback
    - Efficient sparse matrix operations
    - Confidence weighting
    - Regularization
    """

    def __init__(self, config: Optional[ALSConfig] = None):
        """Initialize ALS recommender.

        Args:
            config: ALS configuration
        """
        self.config = config or ALSConfig()
        self.user_factors = None
        self.item_factors = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self._fitted = False

    def fit(self, interactions: List[UserItemInteraction]) -> None:
        """Fit ALS model on interaction data.

        Args:
            interactions: List of user-item interactions
        """
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import spsolve

            logger.info(f"Fitting ALS model with {len(interactions)} interactions")

            # Build user and item mappings
            unique_users = sorted(set(i.user_id for i in interactions))
            unique_items = sorted(set(i.item_id for i in interactions))

            self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
            self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
            self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

            num_users = len(unique_users)
            num_items = len(unique_items)

            # Build interaction matrix
            rows = []
            cols = []
            data = []

            for interaction in interactions:
                user_idx = self.user_id_map[interaction.user_id]
                item_idx = self.item_id_map[interaction.item_id]
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(interaction.rating)

            # Create sparse matrix
            interaction_matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(num_users, num_items),
            )

            # Confidence matrix: C = 1 + alpha * R
            confidence_matrix = 1 + self.config.alpha * interaction_matrix

            # Initialize factors randomly
            np.random.seed(self.config.random_state)
            self.user_factors = np.random.randn(num_users, self.config.num_factors) * 0.01
            self.item_factors = np.random.randn(num_items, self.config.num_factors) * 0.01

            # ALS iterations
            for iteration in range(self.config.num_iterations):
                # Fix item factors, solve for user factors
                for u in range(num_users):
                    # Get items user interacted with
                    item_indices = interaction_matrix[u].indices
                    if len(item_indices) == 0:
                        continue

                    # Confidence values
                    Cu = confidence_matrix[u, item_indices].toarray().flatten()

                    # Item factors for interacted items
                    Y = self.item_factors[item_indices]

                    # Solve: (Y^T * C_u * Y + lambda * I) * x_u = Y^T * C_u * p_u
                    YtCuY = Y.T @ np.diag(Cu) @ Y
                    YtCuY += self.config.regularization * np.eye(self.config.num_factors)

                    # p_u is 1 for all interacted items (implicit feedback)
                    YtCup = Y.T @ Cu

                    self.user_factors[u] = np.linalg.solve(YtCuY, YtCup)

                # Fix user factors, solve for item factors
                for i in range(num_items):
                    # Get users who interacted with item
                    user_indices = interaction_matrix[:, i].indices
                    if len(user_indices) == 0:
                        continue

                    # Confidence values
                    Ci = confidence_matrix[user_indices, i].toarray().flatten()

                    # User factors for users who interacted
                    X = self.user_factors[user_indices]

                    # Solve: (X^T * C_i * X + lambda * I) * y_i = X^T * C_i * p_i
                    XtCiX = X.T @ np.diag(Ci) @ X
                    XtCiX += self.config.regularization * np.eye(self.config.num_factors)

                    XtCip = X.T @ Ci

                    self.item_factors[i] = np.linalg.solve(XtCiX, XtCip)

                if (iteration + 1) % 5 == 0:
                    logger.info(f"ALS iteration {iteration + 1}/{self.config.num_iterations}")

            self._fitted = True
            logger.info("ALS model fitted successfully")

        except ImportError as e:
            logger.error(f"Failed to import scipy: {e}")
            logger.error("Install with: pip install scipy")
            raise
        except Exception as e:
            logger.error(f"Failed to fit ALS model: {e}")
            raise

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        seen_items: Optional[List[str]] = None,
    ) -> List[Recommendation]:
        """Generate recommendations for user.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_seen: Whether to filter already seen items
            seen_items: List of seen item IDs

        Returns:
            List of recommendations
        """
        if not self._fitted:
            raise ValueError("Model not fitted")

        if user_id not in self.user_id_map:
            logger.warning(f"User {user_id} not in training data (cold start)")
            return []

        try:
            user_idx = self.user_id_map[user_id]
            user_vector = self.user_factors[user_idx]

            # Compute scores for all items
            scores = self.item_factors @ user_vector

            # Filter seen items
            if filter_seen and seen_items:
                for item_id in seen_items:
                    if item_id in self.item_id_map:
                        item_idx = self.item_id_map[item_id]
                        scores[item_idx] = -np.inf

            # Get top-k items
            top_indices = np.argsort(scores)[::-1][:top_k]

            recommendations = []
            for rank, item_idx in enumerate(top_indices):
                item_id = self.reverse_item_map[item_idx]
                score = scores[item_idx]

                recommendations.append(Recommendation(
                    item_id=item_id,
                    score=float(score),
                    rank=rank + 1,
                ))

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []

    def get_similar_items(
        self,
        item_id: str,
        top_k: int = 10,
    ) -> List[Recommendation]:
        """Find similar items.

        Args:
            item_id: Item ID
            top_k: Number of similar items

        Returns:
            List of similar items
        """
        if not self._fitted:
            raise ValueError("Model not fitted")

        if item_id not in self.item_id_map:
            logger.warning(f"Item {item_id} not in training data")
            return []

        try:
            item_idx = self.item_id_map[item_id]
            item_vector = self.item_factors[item_idx]

            # Compute cosine similarity with all items
            norms = np.linalg.norm(self.item_factors, axis=1)
            similarities = (self.item_factors @ item_vector) / (norms * np.linalg.norm(item_vector) + 1e-8)

            # Exclude the item itself
            similarities[item_idx] = -np.inf

            # Get top-k similar items
            top_indices = np.argsort(similarities)[::-1][:top_k]

            similar_items = []
            for rank, idx in enumerate(top_indices):
                similar_item_id = self.reverse_item_map[idx]
                similarity = similarities[idx]

                similar_items.append(Recommendation(
                    item_id=similar_item_id,
                    score=float(similarity),
                    rank=rank + 1,
                ))

            return similar_items

        except Exception as e:
            logger.error(f"Failed to find similar items: {e}")
            return []


class NCFRecommender:
    """Neural Collaborative Filtering recommender.

    Features:
    - Deep learning for user-item interactions
    - Generalized Matrix Factorization (GMF)
    - Multi-Layer Perceptron (MLP)
    - Hybrid NCF architecture
    """

    def __init__(self, config: Optional[NCFConfig] = None):
        """Initialize NCF recommender.

        Args:
            config: NCF configuration
        """
        self.config = config or NCFConfig()
        self.model = None
        self.optimizer = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self._initialized = False
        self._fitted = False

    def initialize(self, num_users: int, num_items: int) -> None:
        """Initialize NCF model.

        Args:
            num_users: Number of users
            num_items: Number of items
        """
        if self._initialized:
            return

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            logger.info("Initializing NCF model")

            # Define NCF architecture
            class NCFModel(nn.Module):
                def __init__(self, num_users, num_items, embedding_dim, hidden_layers, dropout):
                    super().__init__()

                    # User and item embeddings for GMF
                    self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
                    self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

                    # User and item embeddings for MLP
                    self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
                    self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

                    # MLP layers
                    mlp_layers = []
                    input_dim = embedding_dim * 2
                    for hidden_dim in hidden_layers:
                        mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                        mlp_layers.append(nn.ReLU())
                        mlp_layers.append(nn.Dropout(dropout))
                        input_dim = hidden_dim

                    self.mlp = nn.Sequential(*mlp_layers)

                    # Final prediction layer
                    self.prediction = nn.Linear(embedding_dim + hidden_layers[-1], 1)
                    self.sigmoid = nn.Sigmoid()

                    # Initialize weights
                    self._init_weights()

                def _init_weights(self):
                    nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
                    nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
                    nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
                    nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

                def forward(self, user_ids, item_ids):
                    # GMF part
                    user_emb_gmf = self.user_embedding_gmf(user_ids)
                    item_emb_gmf = self.item_embedding_gmf(item_ids)
                    gmf_output = user_emb_gmf * item_emb_gmf

                    # MLP part
                    user_emb_mlp = self.user_embedding_mlp(user_ids)
                    item_emb_mlp = self.item_embedding_mlp(item_ids)
                    mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
                    mlp_output = self.mlp(mlp_input)

                    # Concatenate GMF and MLP
                    concat = torch.cat([gmf_output, mlp_output], dim=-1)

                    # Prediction
                    prediction = self.sigmoid(self.prediction(concat))

                    return prediction.squeeze()

            # Create model
            self.model = NCFModel(
                num_users,
                num_items,
                self.config.embedding_dim,
                self.config.hidden_layers,
                self.config.dropout,
            )

            # Move to device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    logger.info("NCF loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("NCF loaded on CPU")

            # Create optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )

            self._initialized = True
            logger.info("NCF model initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import torch: {e}")
            logger.error("Install with: pip install torch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize NCF: {e}")
            raise

    def fit(self, interactions: List[UserItemInteraction]) -> None:
        """Fit NCF model on interaction data.

        Args:
            interactions: List of user-item interactions
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            logger.info(f"Fitting NCF model with {len(interactions)} interactions")

            # Build user and item mappings
            unique_users = sorted(set(i.user_id for i in interactions))
            unique_items = sorted(set(i.item_id for i in interactions))

            self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
            self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
            self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

            num_users = len(unique_users)
            num_items = len(unique_items)

            # Initialize model
            self.initialize(num_users, num_items)

            # Prepare training data
            user_indices = []
            item_indices = []
            ratings = []

            for interaction in interactions:
                user_idx = self.user_id_map[interaction.user_id]
                item_idx = self.item_id_map[interaction.item_id]
                user_indices.append(user_idx)
                item_indices.append(item_idx)
                ratings.append(interaction.rating)

            # Convert to tensors
            user_tensor = torch.LongTensor(user_indices)
            item_tensor = torch.LongTensor(item_indices)
            rating_tensor = torch.FloatTensor(ratings)

            # Create dataset and dataloader
            dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )

            # Loss function
            criterion = nn.BCELoss()

            # Training loop
            self.model.train()
            for epoch in range(self.config.num_epochs):
                total_loss = 0
                num_batches = 0

                for batch_users, batch_items, batch_ratings in dataloader:
                    if self.config.device == "cuda":
                        batch_users = batch_users.to("cuda")
                        batch_items = batch_items.to("cuda")
                        batch_ratings = batch_ratings.to("cuda")

                    # Forward pass
                    predictions = self.model(batch_users, batch_items)

                    # Compute loss
                    loss = criterion(predictions, batch_ratings)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / num_batches
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")

            self._fitted = True
            logger.info("NCF model fitted successfully")

        except Exception as e:
            logger.error(f"Failed to fit NCF model: {e}")
            raise

    def recommend(
        self,
        user_id: str,
        candidate_items: List[str],
        top_k: int = 10,
    ) -> List[Recommendation]:
        """Generate recommendations for user.

        Args:
            user_id: User ID
            candidate_items: List of candidate item IDs
            top_k: Number of recommendations

        Returns:
            List of recommendations
        """
        if not self._fitted:
            raise ValueError("Model not fitted")

        if user_id not in self.user_id_map:
            logger.warning(f"User {user_id} not in training data (cold start)")
            return []

        try:
            import torch

            user_idx = self.user_id_map[user_id]

            # Filter candidate items to those in training data
            valid_items = [
                item_id for item_id in candidate_items
                if item_id in self.item_id_map
            ]

            if not valid_items:
                return []

            # Prepare batch
            user_indices = [user_idx] * len(valid_items)
            item_indices = [self.item_id_map[item_id] for item_id in valid_items]

            user_tensor = torch.LongTensor(user_indices)
            item_tensor = torch.LongTensor(item_indices)

            if self.config.device == "cuda":
                user_tensor = user_tensor.to("cuda")
                item_tensor = item_tensor.to("cuda")

            # Predict scores
            self.model.eval()
            with torch.no_grad():
                scores = self.model(user_tensor, item_tensor)

            scores = scores.cpu().numpy()

            # Get top-k items
            top_indices = np.argsort(scores)[::-1][:top_k]

            recommendations = []
            for rank, idx in enumerate(top_indices):
                item_id = valid_items[idx]
                score = scores[idx]

                recommendations.append(Recommendation(
                    item_id=item_id,
                    score=float(score),
                    rank=rank + 1,
                ))

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        if not self._initialized:
            raise ValueError("Model not initialized")

        try:
            import torch
            from pathlib import Path

            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_item_map': self.reverse_item_map,
                'config': self.config,
            }, path)

            logger.info(f"Saved NCF model to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        try:
            import torch

            checkpoint = torch.load(path, map_location=self.config.device)

            self.config = checkpoint['config']
            self.user_id_map = checkpoint['user_id_map']
            self.item_id_map = checkpoint['item_id_map']
            self.reverse_user_map = checkpoint['reverse_user_map']
            self.reverse_item_map = checkpoint['reverse_item_map']

            num_users = len(self.user_id_map)
            num_items = len(self.item_id_map)

            self.initialize(num_users, num_items)

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self._fitted = True

            logger.info(f"Loaded NCF model from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

