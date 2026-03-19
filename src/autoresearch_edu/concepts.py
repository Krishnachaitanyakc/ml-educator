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
    "transformer": Concept(
        name="transformer",
        definition="A neural network architecture based on self-attention mechanisms, processing all positions in parallel.",
        intuition="Instead of reading a sentence word by word, a transformer looks at the whole sentence at once and figures out how each word relates to every other word.",
        math="Output = LayerNorm(x + MultiHeadAttention(x)) followed by LayerNorm(x + FFN(x))",
        common_mistakes=[
            "Ignoring the quadratic memory cost of self-attention",
            "Not using proper positional encoding",
            "Underestimating the data requirements for training transformers",
        ],
        related_concepts=["self_attention", "positional_encoding", "normalization"],
        explanations={
            "beginner": "A transformer is a type of neural network that can look at all parts of its input at the same time. This makes it great for understanding language, images, and more.",
            "intermediate": "Transformers use self-attention to model dependencies regardless of distance in the input. Each layer has a multi-head attention sublayer and a feed-forward sublayer, both with residual connections and layer normalization. They parallelize better than RNNs during training.",
            "advanced": "The transformer architecture (Vaswani et al., 2017) replaced recurrence with self-attention, enabling O(1) sequential operations at the cost of O(n^2) attention complexity. Scaling laws show predictable loss improvements with model size, data, and compute. Variants include encoder-only (BERT), decoder-only (GPT), and encoder-decoder architectures.",
        },
    ),
    "self_attention": Concept(
        name="self_attention",
        definition="An attention mechanism where queries, keys, and values all come from the same sequence.",
        intuition="Like each word in a sentence looking at every other word to understand context. The word 'it' looks at surrounding words to figure out what 'it' refers to.",
        math="SelfAttention(X) = softmax(XW_Q (XW_K)^T / sqrt(d_k)) * XW_V",
        common_mistakes=[
            "Confusing self-attention with cross-attention",
            "Not using multi-head attention for richer representations",
            "Forgetting the scaling factor sqrt(d_k)",
        ],
        related_concepts=["attention", "transformer", "positional_encoding"],
        explanations={
            "beginner": "Self-attention lets each part of the input pay attention to every other part. It helps the model understand which words or features are related to each other.",
            "intermediate": "Self-attention computes attention weights within a single sequence. Each position generates query, key, and value vectors through learned projections. Multi-head self-attention runs several attention functions in parallel, each learning different relationship patterns.",
            "advanced": "Self-attention provides a fully-connected computation graph over sequence positions with learned, input-dependent weights. It enables gradient flow across arbitrary distances in O(1) layers. Causal masking creates autoregressive variants. Efficient implementations like FlashAttention optimize the memory-bound softmax computation.",
        },
    ),
    "positional_encoding": Concept(
        name="positional_encoding",
        definition="A mechanism to inject sequence order information into transformer models, which are otherwise position-invariant.",
        intuition="Since transformers look at all words at once, positional encoding is like numbering the words so the model knows their order -- like page numbers in a book.",
        math="PE(pos, 2i) = sin(pos / 10000^(2i/d)); PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
        common_mistakes=[
            "Using absolute positional encodings for tasks requiring length generalization",
            "Not considering rotary or relative positional encodings for modern architectures",
            "Confusing learned vs sinusoidal positional encodings",
        ],
        related_concepts=["transformer", "self_attention", "embedding"],
        explanations={
            "beginner": "Positional encoding tells the transformer where each word is in the sentence. Without it, the model would treat 'dog bites man' and 'man bites dog' the same way.",
            "intermediate": "Positional encodings add order information to token embeddings. Sinusoidal encodings use fixed sin/cos functions at different frequencies. Learned encodings are trained parameters. Relative positional encodings (like RoPE) encode distances between positions rather than absolute positions.",
            "advanced": "Positional encodings break the permutation equivariance of self-attention. Sinusoidal encodings theoretically allow extrapolation via inner products encoding relative positions. Rotary Position Embeddings (RoPE) encode positions through rotation matrices applied to Q and K. ALiBi adds linear biases to attention scores proportional to distance.",
        },
    ),
    "gan": Concept(
        name="gan",
        definition="Generative Adversarial Network: two networks (generator and discriminator) trained in opposition to generate realistic data.",
        intuition="Like a forger (generator) trying to create fake paintings and an art expert (discriminator) trying to spot fakes. Both get better over time until the fakes are indistinguishable from real art.",
        math="min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]",
        common_mistakes=[
            "Mode collapse where the generator produces limited variety",
            "Training instability from imbalanced generator/discriminator",
            "Not using techniques like spectral normalization or progressive growing",
        ],
        related_concepts=["discriminator", "loss_function", "regularization"],
        explanations={
            "beginner": "A GAN has two parts that compete: one creates fake data and the other tries to tell real from fake. Through this competition, the creator learns to make very realistic data.",
            "intermediate": "GANs optimize a minimax objective where the generator minimizes and discriminator maximizes the probability of correct classification. Training requires careful balancing. Variants like WGAN use Wasserstein distance for more stable training. Applications include image synthesis, style transfer, and data augmentation.",
            "advanced": "GANs implicitly minimize an f-divergence between real and generated distributions. The original formulation minimizes Jensen-Shannon divergence. Training dynamics can be analyzed as a two-player game; Nash equilibrium corresponds to the generator matching the data distribution. Spectral normalization, gradient penalties, and progressive growing address training instability.",
        },
    ),
    "discriminator": Concept(
        name="discriminator",
        definition="The component in a GAN that learns to distinguish real data from generated (fake) data.",
        intuition="The discriminator is like a detective trying to tell real evidence from planted evidence. It gets better at spotting fakes, which forces the generator to improve.",
        math="D(x) = sigmoid(f(x)), where f is a learned function outputting real/fake logit",
        common_mistakes=[
            "Making the discriminator too powerful relative to the generator",
            "Not using appropriate architecture (e.g., PatchGAN for images)",
            "Ignoring discriminator overfitting on small datasets",
        ],
        related_concepts=["gan", "loss_function", "regularization"],
        explanations={
            "beginner": "The discriminator is the 'judge' in a GAN. It looks at data and decides if it is real or fake. As it gets better at judging, the generator has to get better at creating.",
            "intermediate": "The discriminator is a classifier trained to output high probability for real data and low for generated data. Its gradients guide generator improvement. Techniques like spectral normalization and gradient penalty stabilize discriminator training.",
            "advanced": "The discriminator provides a learned loss function for the generator. In optimal conditions, the discriminator estimates the density ratio p_data/p_g. The discriminator's capacity relative to the generator affects training dynamics and convergence. Feature matching and minibatch discrimination address mode collapse by enriching discriminator feedback.",
        },
    ),
    "policy_gradient": Concept(
        name="policy_gradient",
        definition="A reinforcement learning method that directly optimizes the policy by estimating gradients of expected reward.",
        intuition="Instead of figuring out the value of every possible action, policy gradient methods directly adjust the strategy to do more of what works and less of what does not.",
        math="nabla J(theta) = E[nabla log pi(a|s; theta) * R]",
        common_mistakes=[
            "High variance in gradient estimates without baselines",
            "Not using advantage estimation (GAE) to reduce variance",
            "Ignoring the credit assignment problem in long episodes",
        ],
        related_concepts=["reward_shaping", "loss_function", "optimizer"],
        explanations={
            "beginner": "Policy gradient is a way for an AI agent to learn by trial and error. It increases the probability of actions that led to good outcomes and decreases the probability of actions that led to bad outcomes.",
            "intermediate": "Policy gradient methods use the REINFORCE algorithm to compute gradients of expected return with respect to policy parameters. Variance reduction through baselines (often a value function) is critical. Actor-critic methods combine policy gradients with value estimation.",
            "advanced": "Policy gradient methods optimize E_tau[R(tau)] by computing nabla_theta E[R] = E[sum_t nabla log pi(a_t|s_t;theta) * A_t], where A_t is the advantage. PPO clips the probability ratio to ensure stable updates. TRPO uses a KL divergence constraint. The policy gradient theorem provides unbiased gradients but high variance necessitates careful baseline and GAE design.",
        },
    ),
    "reward_shaping": Concept(
        name="reward_shaping",
        definition="Modifying the reward signal in reinforcement learning to guide the agent toward desired behavior more efficiently.",
        intuition="Like giving a dog treats not just when it completes a trick, but for small steps in the right direction. It helps the learner understand what to do without waiting until the very end.",
        math="R'(s, a, s') = R(s, a, s') + gamma * Phi(s') - Phi(s), where Phi is a potential function",
        common_mistakes=[
            "Introducing reward shaping that changes the optimal policy",
            "Making shaped rewards so dominant they override the true objective",
            "Not using potential-based shaping to guarantee policy invariance",
        ],
        related_concepts=["policy_gradient", "loss_function", "optimizer"],
        explanations={
            "beginner": "Reward shaping gives extra hints to a learning agent so it can learn faster. Instead of only rewarding the final goal, you reward progress along the way.",
            "intermediate": "Reward shaping adds supplementary reward signals to accelerate learning. Potential-based reward shaping (PBRS) guarantees the optimal policy is preserved. Poorly designed shaping can lead to reward hacking where the agent exploits the shaped reward without achieving the true objective.",
            "advanced": "Potential-based reward shaping F(s,s') = gamma*Phi(s')-Phi(s) is the only additive shaping that preserves optimal policies in MDPs (Ng et al., 1999). In practice, shaping potentials are often derived from domain knowledge or learned value estimates. RLHF uses human preferences as an implicit form of reward shaping for language models.",
        },
    ),
    "embedding": Concept(
        name="embedding",
        definition="A learned dense vector representation that maps discrete tokens or categories into continuous space.",
        intuition="Like placing words on a map where similar words are close together. 'King' and 'queen' would be near each other, while 'king' and 'banana' would be far apart.",
        math="e = W_embed[token_id], where W_embed is a learnable |V| x d matrix",
        common_mistakes=[
            "Using embeddings that are too small to capture the complexity of the vocabulary",
            "Not sharing embeddings between input and output layers when appropriate",
            "Ignoring pretrained embeddings when training data is limited",
        ],
        related_concepts=["tokenization", "transformer", "fine_tuning"],
        explanations={
            "beginner": "Embeddings turn words or items into lists of numbers that capture meaning. Similar items get similar numbers, so the model can understand relationships between them.",
            "intermediate": "Embeddings are dense vector representations learned during training. Word2Vec and GloVe learn static embeddings from co-occurrence statistics. In transformers, token embeddings are learned end-to-end. Embedding dimensions typically range from 128 to 4096 depending on model size.",
            "advanced": "Embeddings define a continuous representation space where algebraic operations approximate semantic relationships (e.g., king - man + woman ~ queen). In transformers, embeddings are often tied between input and output layers (weight tying), reducing parameters. The embedding matrix can be decomposed or quantized for efficiency. Contrastive learning methods learn embeddings by pushing similar pairs together and dissimilar pairs apart.",
        },
    ),
    "tokenization": Concept(
        name="tokenization",
        definition="The process of converting raw text into a sequence of tokens (subwords, words, or characters) for model input.",
        intuition="Like breaking a sentence into puzzle pieces. You could break at every letter, every word, or somewhere in between. The trick is finding pieces that are meaningful and efficient.",
        math=None,
        common_mistakes=[
            "Using a tokenizer not matched to the model's training vocabulary",
            "Ignoring tokenization effects on sequence length and cost",
            "Not handling special tokens (BOS, EOS, PAD) correctly",
        ],
        related_concepts=["embedding", "transformer", "positional_encoding"],
        explanations={
            "beginner": "Tokenization breaks text into smaller pieces that the model can understand. These pieces might be whole words, parts of words, or even single characters.",
            "intermediate": "Modern tokenizers like BPE (Byte Pair Encoding) and SentencePiece learn a vocabulary of subword units from training data. This balances vocabulary size with sequence length. Rare words are split into known subwords. Vocabulary size affects model size (embedding matrix) and sequence length.",
            "advanced": "BPE iteratively merges the most frequent byte pairs to build a vocabulary. Unigram tokenization (SentencePiece) treats tokenization as a probabilistic model selecting the most likely segmentation. Tokenization choices affect downstream performance: morphologically-aware tokenizers can improve multilingual models. Byte-level approaches (ByT5) eliminate the vocabulary bottleneck entirely at the cost of longer sequences.",
        },
    ),
    "fine_tuning": Concept(
        name="fine_tuning",
        definition="Adapting a pretrained model to a specific task or domain by continuing training on task-specific data.",
        intuition="Like a general doctor specializing in cardiology. They already know medicine (pretraining), and now they focus on heart-related knowledge (fine-tuning).",
        math=None,
        common_mistakes=[
            "Using too high a learning rate which destroys pretrained knowledge",
            "Fine-tuning on too little data leading to overfitting",
            "Not freezing early layers when fine-tuning data is very limited",
        ],
        related_concepts=["learning_rate", "regularization", "embedding"],
        explanations={
            "beginner": "Fine-tuning takes a model that already learned general knowledge and trains it a bit more on your specific task. This is much faster and needs less data than training from scratch.",
            "intermediate": "Fine-tuning adapts pretrained representations to downstream tasks. Key decisions include which layers to freeze, learning rate selection (typically 10-100x smaller than pretraining), and training duration. Parameter-efficient methods like LoRA and adapters fine-tune only a small subset of parameters.",
            "advanced": "Fine-tuning performs transfer learning from a pretrained initialization. Catastrophic forgetting can be mitigated through elastic weight consolidation, progressive unfreezing, or discriminative learning rates. LoRA approximates weight updates as low-rank matrices (W + BA where B,A have rank r << d). The lottery ticket hypothesis suggests fine-tuning may activate task-relevant subnetworks within the pretrained model.",
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

    def load_from_yaml(self, path: str) -> int:
        """Load custom concepts from a YAML file. Returns count of concepts loaded."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not data:
            return 0
        concepts = data if isinstance(data, list) else [data]
        count = 0
        for entry in concepts:
            concept = Concept(
                name=entry["name"],
                definition=entry["definition"],
                intuition=entry.get("intuition", ""),
                math=entry.get("math"),
                common_mistakes=entry.get("common_mistakes", []),
                related_concepts=entry.get("related_concepts", []),
                explanations=entry.get("explanations", {}),
            )
            self._concepts[concept.name] = concept
            count += 1
        return count
