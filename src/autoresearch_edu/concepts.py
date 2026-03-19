"""ML concept library with multi-level explanations."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Concept:
    """An ML concept with explanations at different levels."""
    name: str
    definition: str
    intuition: str
    math: Optional[str]
    common_mistakes: List[str]
    related_concepts: List[str]
    explanations: Dict[str, str] = field(default_factory=dict)

    def get_explanation(self, level: str) -> str:
        """Get explanation at a given skill level."""
        if level in self.explanations:
            return self.explanations[level]
        return self.definition


_CONCEPTS = {
    "learning_rate": Concept(
        name="learning_rate",
        definition="The step size used to update model weights during training.",
        intuition="Think of it as how big of steps you take when walking downhill. Too big and you overshoot the bottom, too small and you take forever to get there.",
        math="w_new = w_old - lr * gradient",
        common_mistakes=[
            "Setting learning rate too high causes training to diverge",
            "Setting it too low leads to very slow convergence",
            "Not using a learning rate schedule",
        ],
        related_concepts=["optimizer", "loss_function", "batch_size"],
        explanations={
            "beginner": "The learning rate controls how quickly your model learns. A bigger learning rate means faster but rougher learning, while a smaller one means slower but more careful learning.",
            "intermediate": "The learning rate is a hyperparameter that controls the magnitude of weight updates. It directly scales the gradient during optimization. Common values range from 1e-4 to 1e-2, and schedules like cosine annealing or warm-up are often used.",
            "advanced": "The learning rate determines the step size in gradient-based optimization: w_{t+1} = w_t - eta * nabla L(w_t). The choice interacts with batch size (linear scaling rule), optimizer state (Adam adapts per-parameter), and loss landscape geometry. Critical learning rate theory suggests phase transitions in training dynamics.",
        },
    ),
    "batch_size": Concept(
        name="batch_size",
        definition="The number of training examples used in one iteration of model update.",
        intuition="Imagine grading exams. You could grade them one at a time (batch size 1) or look at all of them at once. Batch size is how many you look at before making a decision about what to teach next.",
        math="gradient = (1/B) * sum(gradient_i for i in batch)",
        common_mistakes=[
            "Using batch size too large can lead to poor generalization",
            "Very small batch sizes increase training noise and time",
            "Not adjusting learning rate when changing batch size",
        ],
        related_concepts=["learning_rate", "optimizer", "regularization"],
        explanations={
            "beginner": "Batch size is how many examples your model looks at before updating itself. More examples at once gives a clearer picture but uses more memory.",
            "intermediate": "Batch size affects both the noise in gradient estimates and computational efficiency. Larger batches give more stable gradients but may converge to sharper minima. The learning rate should typically be scaled with batch size (linear scaling rule).",
            "advanced": "Batch size controls the signal-to-noise ratio of gradient estimates. Under the linear scaling rule, lr should scale linearly with batch size. Large-batch training often requires warm-up schedules. The generalization gap between small and large batch training relates to the sharpness of minima (SAM, SWA help address this).",
        },
    ),
    "regularization": Concept(
        name="regularization",
        definition="Techniques to prevent overfitting by adding constraints or penalties to the model.",
        intuition="Like adding rules to a game to keep it fair. Without rules, a model might memorize the answers instead of actually learning the patterns.",
        math="L_total = L_data + lambda * R(w), where R(w) is the regularization term",
        common_mistakes=[
            "Too much regularization causes underfitting",
            "Not using any regularization on complex models",
            "Applying dropout during evaluation/inference",
        ],
        related_concepts=["loss_function", "architecture_depth", "batch_size"],
        explanations={
            "beginner": "Regularization is like giving your model guardrails. It prevents the model from just memorizing training data and helps it learn general patterns that work on new data too.",
            "intermediate": "Regularization techniques (L1, L2, dropout, weight decay, data augmentation) add constraints that improve generalization. L2 regularization penalizes large weights, dropout randomly disables neurons during training, and data augmentation increases effective dataset size.",
            "advanced": "Regularization modifies the optimization landscape to favor solutions with better generalization. L2 regularization (weight decay) biases toward smaller-norm solutions. Dropout provides an approximate Bayesian ensemble. The interplay between implicit regularization (SGD noise, architecture choices) and explicit regularization is an active research area.",
        },
    ),
    "attention": Concept(
        name="attention",
        definition="A mechanism that allows models to focus on relevant parts of the input when producing output.",
        intuition="Like reading a book and highlighting the most important parts. Attention lets the model decide which parts of the input are most relevant for each part of the output.",
        math="Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V",
        common_mistakes=[
            "Not scaling attention scores by sqrt(d_k)",
            "Ignoring memory requirements of full attention (quadratic in sequence length)",
            "Confusing self-attention with cross-attention",
        ],
        related_concepts=["normalization", "architecture_depth", "loss_function"],
        explanations={
            "beginner": "Attention is how a model decides what to focus on. Just like when you read a sentence, some words are more important than others for understanding the meaning.",
            "intermediate": "Attention computes weighted sums of values, where weights are determined by the compatibility of queries and keys. Multi-head attention allows the model to attend to information from different representation subspaces. Self-attention processes the input by relating different positions within the same sequence.",
            "advanced": "The scaled dot-product attention Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V has O(n^2) complexity in sequence length. Efficient variants (linear attention, sparse attention, FlashAttention) address this. The attention mechanism can be viewed as a soft dictionary lookup or kernel smoother. Recent work explores attention as a key-value memory system.",
        },
    ),
    "normalization": Concept(
        name="normalization",
        definition="Techniques to standardize intermediate representations in a neural network.",
        intuition="Like adjusting the volume on different audio tracks so they all play at a similar level. It keeps the numbers flowing through the network in a reasonable range.",
        math="y = (x - mean) / sqrt(var + epsilon) * gamma + beta",
        common_mistakes=[
            "Using batch normalization with very small batch sizes",
            "Forgetting that batch norm behaves differently during training vs inference",
            "Not including the learnable affine parameters (gamma, beta)",
        ],
        related_concepts=["batch_size", "architecture_depth", "learning_rate"],
        explanations={
            "beginner": "Normalization keeps the numbers inside your model from getting too big or too small. This helps the model train more smoothly and quickly.",
            "intermediate": "Normalization layers (BatchNorm, LayerNorm, GroupNorm) stabilize training by reducing internal covariate shift. LayerNorm normalizes across features for each sample, making it independent of batch size, which is why it is preferred in transformers.",
            "advanced": "Normalization addresses the problem of internal covariate shift and smooths the loss landscape. BatchNorm introduces implicit regularization through batch statistics. LayerNorm is invariant to re-scaling of weights and inputs. The interaction between normalization and residual connections is crucial for training very deep networks.",
        },
    ),
    "optimizer": Concept(
        name="optimizer",
        definition="The algorithm used to update model weights based on computed gradients.",
        intuition="The optimizer is like your strategy for finding the lowest point in a hilly landscape. SGD walks straight downhill, while Adam is smarter and adjusts its speed based on the terrain.",
        math="SGD: w = w - lr * grad; Adam: w = w - lr * m_hat / (sqrt(v_hat) + eps)",
        common_mistakes=[
            "Using SGD without momentum for complex architectures",
            "Not tuning optimizer-specific hyperparameters (betas for Adam)",
            "Ignoring the interaction between optimizer choice and learning rate",
        ],
        related_concepts=["learning_rate", "loss_function", "batch_size"],
        explanations={
            "beginner": "The optimizer is the method your model uses to improve itself. Think of it as different strategies for finding the best answer -- some are simple, some are clever about adapting their approach.",
            "intermediate": "Optimizers like Adam, SGD with momentum, and AdamW differ in how they use gradient information. Adam maintains per-parameter adaptive learning rates using first and second moment estimates. AdamW decouples weight decay from the adaptive learning rate.",
            "advanced": "Optimizer choice interacts with loss landscape geometry. Adam's adaptive learning rates can lead to poor generalization vs SGD in some settings. The effective learning rate in Adam is lr/sqrt(v), providing implicit gradient clipping. Techniques like LARS/LAMB extend adaptive methods for large-batch distributed training.",
        },
    ),
    "loss_function": Concept(
        name="loss_function",
        definition="A function that measures how well the model's predictions match the target values.",
        intuition="The loss function is like a score that tells you how wrong your model is. Lower is better. The training process tries to make this score as low as possible.",
        math="Cross-entropy: L = -sum(y_true * log(y_pred)); MSE: L = mean((y_true - y_pred)^2)",
        common_mistakes=[
            "Using MSE loss for classification tasks",
            "Not handling class imbalance in the loss function",
            "Ignoring numerical stability (log(0) issues)",
        ],
        related_concepts=["optimizer", "regularization", "learning_rate"],
        explanations={
            "beginner": "The loss function is how we measure mistakes. Cross-entropy is used when predicting categories (like cat vs dog), and MSE is used when predicting numbers (like temperature).",
            "intermediate": "Loss functions define the optimization objective. Cross-entropy loss for classification naturally handles probability outputs from softmax. For imbalanced datasets, focal loss or weighted cross-entropy can help. The choice of loss affects gradient magnitudes and training dynamics.",
            "advanced": "The loss function defines the optimization landscape. Cross-entropy is the negative log-likelihood under a categorical distribution. Label smoothing modifies the target distribution to improve calibration. Contrastive losses (InfoNCE, triplet) operate on representation geometry rather than direct prediction.",
        },
    ),
    "architecture_depth": Concept(
        name="architecture_depth",
        definition="The number of layers in a neural network, determining its representational capacity.",
        intuition="Think of depth like the number of steps in a recipe. More steps let you create more complex dishes, but too many steps make the recipe hard to follow and things can go wrong.",
        math=None,
        common_mistakes=[
            "Making the network too deep without residual connections",
            "Not accounting for vanishing/exploding gradients in deep networks",
            "Assuming deeper is always better without considering computation cost",
        ],
        related_concepts=["normalization", "regularization", "attention"],
        explanations={
            "beginner": "Architecture depth is how many layers your neural network has. More layers can learn more complex patterns, but they also need more data and compute, and can be harder to train.",
            "intermediate": "Deeper networks can represent more complex functions but suffer from optimization challenges. Residual connections (skip connections) enable training very deep networks by providing gradient shortcuts. The effective depth of a network can differ from its nominal depth.",
            "advanced": "Network depth relates to the compositional complexity of learned functions. The residual connection f(x) + x creates an ensemble of paths of different lengths. Neural architecture search (NAS) can automatically find optimal depth configurations. Recent work on scaling laws provides empirical guidance on depth vs width tradeoffs.",
        },
    ),
}


class ConceptLibrary:
    """Library of ML concepts with multi-level explanations."""

    def __init__(self):
        self._concepts = dict(_CONCEPTS)

    def get(self, name: str) -> Concept:
        """Get a concept by name."""
        if name not in self._concepts:
            raise KeyError(f"Unknown concept: {name}")
        return self._concepts[name]

    def list_concepts(self) -> List[str]:
        """List all available concept names."""
        return list(self._concepts.keys())

    def search(self, query: str) -> List[Concept]:
        """Search concepts by name or definition."""
        query_lower = query.lower()
        results = []
        for concept in self._concepts.values():
            if (query_lower in concept.name.lower()
                    or query_lower in concept.definition.lower()):
                results.append(concept)
        return results
