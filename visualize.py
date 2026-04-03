# visualize_training_metrics.py
import json
import os
import matplotlib.pyplot as plt


def moving_average(values, window=20):
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        smoothed.append(sum(values[left:i + 1]) / (i - left + 1))
    return smoothed


def main():
    metrics_path = r"C:\Users\86153\Downloads\training_metrics (1).json"
    out_dir = os.path.dirname(metrics_path)

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_loss_items = data.get("train_loss", [])
    val_loss_items = data.get("val_loss", [])

    if not train_loss_items and not val_loss_items:
        raise ValueError("No train_loss/val_loss found in JSON.")

    # 训练曲线（按记录顺序）
    train_steps = list(range(1, len(train_loss_items) + 1))
    train_losses = [x["loss"] for x in train_loss_items]
    train_losses_smooth = moving_average(train_losses, window=20)

    # 验证曲线（按 epoch）
    val_epochs = [x["epoch"] for x in val_loss_items]
    val_losses = [x["val_loss"] for x in val_loss_items]

    plt.figure(figsize=(12, 5))

    # 子图1：train loss
    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(train_steps, train_losses, alpha=0.35, label="Train Loss (raw)")
        plt.plot(train_steps, train_losses_smooth, linewidth=2, label="Train Loss (smoothed)")
    plt.xlabel("Logged Train Step Index")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    # 子图2：val loss
    plt.subplot(1, 2, 2)
    if val_losses:
        plt.plot(val_epochs, val_losses, marker="o", linewidth=2, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    out_png = os.path.join(out_dir, "training_metrics_plot.png")
    plt.savefig(out_png, dpi=200)
    plt.show()

    print(f"Saved figure to: {out_png}")


if __name__ == "__main__":
    main()
