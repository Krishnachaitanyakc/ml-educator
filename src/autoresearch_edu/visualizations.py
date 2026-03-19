"""Generate matplotlib diagrams for ML concepts."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_rate_landscape(output_path: str = "learning_rate.png"):
    """Plot loss landscape showing effect of different learning rates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-3, 3, 300)
    loss = 0.5 * x**2 + 0.3 * np.sin(3 * x)
    ax.plot(x, loss, "k-", linewidth=2, label="Loss landscape")

    # Simulate gradient descent with different learning rates
    for lr, color, label in [(0.01, "green", "Small LR (0.01)"),
                              (0.1, "blue", "Good LR (0.1)"),
                              (0.9, "red", "Large LR (0.9)")]:
        pos = 2.5
        positions = [pos]
        for _ in range(15):
            grad = pos + 0.9 * np.cos(3 * pos)
            pos = pos - lr * grad
            pos = np.clip(pos, -3, 3)
            positions.append(pos)
        ys = [0.5 * p**2 + 0.3 * np.sin(3 * p) for p in positions]
        ax.plot(positions, ys, "o-", color=color, markersize=4, alpha=0.7, label=label)

    ax.set_xlabel("Weight value")
    ax.set_ylabel("Loss")
    ax.set_title("Effect of Learning Rate on Optimization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def plot_attention_heatmap(output_path: str = "attention.png"):
    """Plot an example attention weight heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(tokens)
    np.random.seed(42)
    raw = np.random.randn(n, n)
    # Make some positions attend more to relevant tokens
    raw[1, 0] += 2  # cat attends to The
    raw[2, 1] += 2  # sat attends to cat
    raw[4, 5] += 2  # the attends to mat
    raw[5, 2] += 1.5  # mat attends to sat
    weights = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)

    im = ax.imshow(weights, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title("Self-Attention Weights")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def plot_optimizer_paths(output_path: str = "optimizer.png"):
    """Plot gradient descent paths for different optimizers on a 2D loss surface."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Create a 2D loss surface (elongated bowl)
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * X**2 + 5 * Y**2

    ax.contour(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.6)

    def sgd_step(pos, lr=0.1):
        gx, gy = pos[0], 10 * pos[1]
        return [pos[0] - lr * gx, pos[1] - lr * gy]

    def adam_step(pos, m, v, t, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8):
        gx, gy = pos[0], 10 * pos[1]
        g = np.array([gx, gy])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**(t + 1))
        v_hat = v / (1 - beta2**(t + 1))
        new_pos = [pos[0] - lr * m_hat[0] / (np.sqrt(v_hat[0]) + eps),
                   pos[1] - lr * m_hat[1] / (np.sqrt(v_hat[1]) + eps)]
        return new_pos, m, v

    # SGD path
    pos = [2.5, 2.5]
    sgd_path = [pos[:]]
    for _ in range(30):
        pos = sgd_step(pos)
        sgd_path.append(pos[:])
    sgd_path = np.array(sgd_path)
    ax.plot(sgd_path[:, 0], sgd_path[:, 1], "o-", color="blue", markersize=3, label="SGD", alpha=0.8)

    # Adam path
    pos = [2.5, 2.5]
    m, v = np.zeros(2), np.zeros(2)
    adam_path = [pos[:]]
    for t in range(30):
        pos, m, v = adam_step(pos, m, v, t)
        adam_path.append(pos[:])
    adam_path = np.array(adam_path)
    ax.plot(adam_path[:, 0], adam_path[:, 1], "s-", color="red", markersize=3, label="Adam", alpha=0.8)

    ax.plot(0, 0, "*", color="gold", markersize=15, label="Minimum")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("Optimizer Paths on Loss Surface")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


DIAGRAM_GENERATORS = {
    "learning_rate": plot_learning_rate_landscape,
    "attention": plot_attention_heatmap,
    "self_attention": plot_attention_heatmap,
    "optimizer": plot_optimizer_paths,
}
