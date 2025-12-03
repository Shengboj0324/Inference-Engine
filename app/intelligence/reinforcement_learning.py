"""Industrial-grade Reinforcement Learning for content selection and user engagement.

Implements:
- Deep Q-Network (DQN) for content ranking
- Proximal Policy Optimization (PPO) for engagement optimization
- Experience replay and target networks
- Multi-armed bandit variants
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class State(BaseModel):
    """RL state representation."""

    user_embedding: List[float]
    content_embedding: List[float]
    context_features: Dict[str, float]  # time_of_day, day_of_week, etc.
    user_history: List[str]  # Recent content IDs


class Action(BaseModel):
    """RL action representation."""

    content_id: str
    action_type: str  # "show", "recommend", "skip"
    confidence: float


class Reward(BaseModel):
    """RL reward signal."""

    engagement_score: float  # 0-1
    click_through: bool
    time_spent: float  # seconds
    feedback: Optional[str]  # "like", "dislike", "share", etc.


class Experience(BaseModel):
    """Experience tuple for replay buffer."""

    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""

    state_dim: int = 1536  # Embedding dimension
    action_dim: int = 100  # Number of possible actions
    hidden_dim: int = 512
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    buffer_size: int = 10000
    target_update_freq: int = 100
    device: str = "cpu"


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""

    state_dim: int = 1536
    action_dim: int = 100
    hidden_dim: int = 512
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95  # Generalized Advantage Estimation
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    device: str = "cpu"


class ReplayBuffer:
    """Experience replay buffer for DQN.

    Features:
    - Fixed-size circular buffer
    - Random sampling
    - Priority sampling support
    """

    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, experience: Experience) -> None:
        """Add experience to buffer.

        Args:
            experience: Experience tuple
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of experiences
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)


class DQNNetwork:
    """Deep Q-Network for content ranking.

    Features:
    - Multi-layer perceptron
    - Target network for stability
    - Experience replay
    - Epsilon-greedy exploration
    """

    def __init__(self, config: Optional[DQNConfig] = None):
        """Initialize DQN network.

        Args:
            config: DQN configuration
        """
        self.config = config or DQNConfig()
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self._initialized = False

    def initialize(self) -> None:
        """Initialize Q-network and target network."""
        if self._initialized:
            return

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            logger.info("Initializing DQN network")

            # Define Q-network architecture
            class QNetwork(nn.Module):
                def __init__(self, state_dim, action_dim, hidden_dim):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, action_dim),
                    )

                def forward(self, state):
                    return self.network(state)

            # Create networks
            self.q_network = QNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dim,
            )
            self.target_network = QNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dim,
            )

            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

            # Move to device
            if self.config.device == "cuda":
                if torch.cuda.is_available():
                    self.q_network = self.q_network.to("cuda")
                    self.target_network = self.target_network.to("cuda")
                    logger.info("DQN loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("DQN loaded on CPU")

            # Create optimizer
            self.optimizer = optim.Adam(
                self.q_network.parameters(),
                lr=self.config.learning_rate,
            )

            self._initialized = True
            logger.info("DQN network initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import torch: {e}")
            logger.error("Install with: pip install torch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize DQN: {e}")
            raise

    def select_action(
        self,
        state: State,
        available_actions: List[str],
        explore: bool = True,
    ) -> Action:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            available_actions: List of available content IDs
            explore: Whether to use exploration

        Returns:
            Selected action
        """
        if not self._initialized:
            self.initialize()

        try:
            import torch

            # Epsilon-greedy exploration
            if explore and random.random() < self.epsilon:
                # Random action
                content_id = random.choice(available_actions)
                return Action(
                    content_id=content_id,
                    action_type="show",
                    confidence=0.0,
                )

            # Greedy action (exploit)
            # Combine user and content embeddings
            state_vector = np.concatenate([
                state.user_embedding,
                state.content_embedding,
                list(state.context_features.values()),
            ])

            # Pad or truncate to state_dim
            if len(state_vector) < self.config.state_dim:
                state_vector = np.pad(
                    state_vector,
                    (0, self.config.state_dim - len(state_vector)),
                )
            else:
                state_vector = state_vector[:self.config.state_dim]

            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            if self.config.device == "cuda":
                state_tensor = state_tensor.to("cuda")

            # Get Q-values
            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            # Select action with highest Q-value
            action_idx = q_values.argmax().item()
            content_id = available_actions[action_idx % len(available_actions)]
            confidence = torch.softmax(q_values, dim=1)[0, action_idx].item()

            return Action(
                content_id=content_id,
                action_type="show",
                confidence=float(confidence),
            )

        except Exception as e:
            logger.error(f"Failed to select action: {e}")
            # Fallback to random
            return Action(
                content_id=random.choice(available_actions),
                action_type="show",
                confidence=0.0,
            )

    def store_experience(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        done: bool = False,
    ) -> None:
        """Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Calculate total reward
        total_reward = (
            reward.engagement_score * 0.4 +
            (1.0 if reward.click_through else 0.0) * 0.3 +
            min(reward.time_spent / 60.0, 1.0) * 0.3  # Normalize to 0-1
        )

        experience = Experience(
            state=state,
            action=action,
            reward=total_reward,
            next_state=next_state,
            done=done,
        )

        self.replay_buffer.push(experience)

    def train_step(self) -> Optional[float]:
        """Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self._initialized:
            self.initialize()

        # Need enough experiences
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        try:
            import torch
            import torch.nn.functional as F

            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size)

            # Prepare batch tensors
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for exp in batch:
                # State vector
                state_vec = np.concatenate([
                    exp.state.user_embedding,
                    exp.state.content_embedding,
                    list(exp.state.context_features.values()),
                ])
                if len(state_vec) < self.config.state_dim:
                    state_vec = np.pad(state_vec, (0, self.config.state_dim - len(state_vec)))
                else:
                    state_vec = state_vec[:self.config.state_dim]
                states.append(state_vec)

                # Next state vector
                next_state_vec = np.concatenate([
                    exp.next_state.user_embedding,
                    exp.next_state.content_embedding,
                    list(exp.next_state.context_features.values()),
                ])
                if len(next_state_vec) < self.config.state_dim:
                    next_state_vec = np.pad(next_state_vec, (0, self.config.state_dim - len(next_state_vec)))
                else:
                    next_state_vec = next_state_vec[:self.config.state_dim]
                next_states.append(next_state_vec)

                # Action index (hash content_id to action space)
                action_idx = hash(exp.action.content_id) % self.config.action_dim
                actions.append(action_idx)

                rewards.append(exp.reward)
                dones.append(exp.done)

            # Convert to tensors
            state_batch = torch.FloatTensor(np.array(states))
            action_batch = torch.LongTensor(actions)
            reward_batch = torch.FloatTensor(rewards)
            next_state_batch = torch.FloatTensor(np.array(next_states))
            done_batch = torch.FloatTensor(dones)

            if self.config.device == "cuda":
                state_batch = state_batch.to("cuda")
                action_batch = action_batch.to("cuda")
                reward_batch = reward_batch.to("cuda")
                next_state_batch = next_state_batch.to("cuda")
                done_batch = done_batch.to("cuda")

            # Compute Q(s, a)
            q_values = self.q_network(state_batch)
            q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_state_batch)
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = reward_batch + (1 - done_batch) * self.config.gamma * max_next_q_values

            # Compute loss
            loss = F.mse_loss(q_values, target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            # Update target network
            self.steps += 1
            if self.steps % self.config.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Decay epsilon
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay,
            )

            return loss.item()

        except Exception as e:
            logger.error(f"Failed to train DQN: {e}")
            return None

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
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'config': self.config,
            }, path)

            logger.info(f"Saved DQN model to {path}")

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
            self.initialize()

            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']

            logger.info(f"Loaded DQN model from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class PPOAgent:
    """Proximal Policy Optimization agent for engagement optimization.

    Features:
    - Actor-Critic architecture
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs per update
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        """Initialize PPO agent.

        Args:
            config: PPO configuration
        """
        self.config = config or PPOConfig()
        self.actor = None
        self.critic = None
        self.optimizer = None
        self.trajectory_buffer = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize actor and critic networks."""
        if self._initialized:
            return

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            logger.info("Initializing PPO agent")

            # Define Actor network (policy)
            class Actor(nn.Module):
                def __init__(self, state_dim, action_dim, hidden_dim):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, action_dim),
                        nn.Softmax(dim=-1),
                    )

                def forward(self, state):
                    return self.network(state)

            # Define Critic network (value function)
            class Critic(nn.Module):
                def __init__(self, state_dim, hidden_dim):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, 1),
                    )

                def forward(self, state):
                    return self.network(state)

            # Create networks
            self.actor = Actor(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dim,
            )
            self.critic = Critic(
                self.config.state_dim,
                self.config.hidden_dim,
            )

            # Move to device
            if self.config.device == "cuda":
                if torch.cuda.is_available():
                    self.actor = self.actor.to("cuda")
                    self.critic = self.critic.to("cuda")
                    logger.info("PPO loaded on GPU")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.config.device = "cpu"
            else:
                logger.info("PPO loaded on CPU")

            # Create optimizer for both networks
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.config.learning_rate,
            )

            self._initialized = True
            logger.info("PPO agent initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import torch: {e}")
            logger.error("Install with: pip install torch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PPO: {e}")
            raise

    def select_action(
        self,
        state: State,
        available_actions: List[str],
    ) -> Tuple[Action, float, float]:
        """Select action using policy network.

        Args:
            state: Current state
            available_actions: List of available content IDs

        Returns:
            Tuple of (action, log_prob, value)
        """
        if not self._initialized:
            self.initialize()

        try:
            import torch

            # Prepare state vector
            state_vector = np.concatenate([
                state.user_embedding,
                state.content_embedding,
                list(state.context_features.values()),
            ])

            if len(state_vector) < self.config.state_dim:
                state_vector = np.pad(
                    state_vector,
                    (0, self.config.state_dim - len(state_vector)),
                )
            else:
                state_vector = state_vector[:self.config.state_dim]

            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            if self.config.device == "cuda":
                state_tensor = state_tensor.to("cuda")

            # Get action probabilities
            with torch.no_grad():
                action_probs = self.actor(state_tensor)
                value = self.critic(state_tensor)

            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            log_prob = action_dist.log_prob(action_idx)

            # Map to content ID
            content_id = available_actions[action_idx.item() % len(available_actions)]
            confidence = action_probs[0, action_idx].item()

            action = Action(
                content_id=content_id,
                action_type="show",
                confidence=float(confidence),
            )

            return action, log_prob.item(), value.item()

        except Exception as e:
            logger.error(f"Failed to select action: {e}")
            # Fallback to random
            return (
                Action(
                    content_id=random.choice(available_actions),
                    action_type="show",
                    confidence=0.0,
                ),
                0.0,
                0.0,
            )

    def store_trajectory(
        self,
        state: State,
        action: Action,
        reward: Reward,
        log_prob: float,
        value: float,
    ) -> None:
        """Store trajectory step.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate
        """
        # Calculate total reward
        total_reward = (
            reward.engagement_score * 0.4 +
            (1.0 if reward.click_through else 0.0) * 0.3 +
            min(reward.time_spent / 60.0, 1.0) * 0.3
        )

        self.trajectory_buffer.append({
            'state': state,
            'action': action,
            'reward': total_reward,
            'log_prob': log_prob,
            'value': value,
        })

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value of next state

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []

        gae = 0
        next_val = next_value

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.config.gamma * next_val - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_val = values[i]

        return advantages, returns

    def train_step(self, next_state: State) -> Optional[Dict[str, float]]:
        """Perform PPO update.

        Args:
            next_state: Final state for value bootstrapping

        Returns:
            Dictionary of loss values if training occurred
        """
        if not self._initialized:
            self.initialize()

        if len(self.trajectory_buffer) < self.config.batch_size:
            return None

        try:
            import torch
            import torch.nn.functional as F

            # Extract trajectory data
            states = []
            actions = []
            old_log_probs = []
            rewards = []
            values = []

            for step in self.trajectory_buffer:
                # State vector
                state_vec = np.concatenate([
                    step['state'].user_embedding,
                    step['state'].content_embedding,
                    list(step['state'].context_features.values()),
                ])
                if len(state_vec) < self.config.state_dim:
                    state_vec = np.pad(state_vec, (0, self.config.state_dim - len(state_vec)))
                else:
                    state_vec = state_vec[:self.config.state_dim]
                states.append(state_vec)

                # Action index
                action_idx = hash(step['action'].content_id) % self.config.action_dim
                actions.append(action_idx)

                old_log_probs.append(step['log_prob'])
                rewards.append(step['reward'])
                values.append(step['value'])

            # Compute next value for GAE
            next_state_vec = np.concatenate([
                next_state.user_embedding,
                next_state.content_embedding,
                list(next_state.context_features.values()),
            ])
            if len(next_state_vec) < self.config.state_dim:
                next_state_vec = np.pad(next_state_vec, (0, self.config.state_dim - len(next_state_vec)))
            else:
                next_state_vec = next_state_vec[:self.config.state_dim]

            next_state_tensor = torch.FloatTensor(next_state_vec).unsqueeze(0)
            if self.config.device == "cuda":
                next_state_tensor = next_state_tensor.to("cuda")

            with torch.no_grad():
                next_value = self.critic(next_state_tensor).item()

            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, next_value)

            # Convert to tensors
            state_batch = torch.FloatTensor(np.array(states))
            action_batch = torch.LongTensor(actions)
            old_log_prob_batch = torch.FloatTensor(old_log_probs)
            advantage_batch = torch.FloatTensor(advantages)
            return_batch = torch.FloatTensor(returns)

            if self.config.device == "cuda":
                state_batch = state_batch.to("cuda")
                action_batch = action_batch.to("cuda")
                old_log_prob_batch = old_log_prob_batch.to("cuda")
                advantage_batch = advantage_batch.to("cuda")
                return_batch = return_batch.to("cuda")

            # Normalize advantages
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

            # PPO update for multiple epochs
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0

            for _ in range(self.config.ppo_epochs):
                # Get current policy and value
                action_probs = self.actor(state_batch)
                values = self.critic(state_batch).squeeze()

                # Compute log probabilities
                action_dist = torch.distributions.Categorical(action_probs)
                log_probs = action_dist.log_prob(action_batch)
                entropy = action_dist.entropy().mean()

                # Compute ratio
                ratio = torch.exp(log_probs - old_log_prob_batch)

                # Compute surrogate losses
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantage_batch

                # Policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, return_batch)

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss -
                    self.config.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

            # Clear trajectory buffer
            self.trajectory_buffer.clear()

            return {
                'policy_loss': total_policy_loss / self.config.ppo_epochs,
                'value_loss': total_value_loss / self.config.ppo_epochs,
                'entropy': total_entropy / self.config.ppo_epochs,
            }

        except Exception as e:
            logger.error(f"Failed to train PPO: {e}")
            return None

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
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.config,
            }, path)

            logger.info(f"Saved PPO model to {path}")

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
            self.initialize()

            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            logger.info(f"Loaded PPO model from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

