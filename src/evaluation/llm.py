"""Action Predictor — predicts the best HTN action from a screenshot + game state.

Architecture
────────────
  screenshot ──► ResNet-18 (ImageNet, fine-tuned) ──► 512-d ─┐
                                                              ├─► MLP ──► action class
  state (x, y, z, action_counts…) ──► normalised vector ─────┘

Uses a pre-trained ResNet-18 from torchvision as visual backbone.
Early layers are frozen; the last residual block (layer4) is fine-tuned
together with the lightweight classification head.
With ~4 k examples this is far more effective than training features
from scratch or reusing a VAE encoder trained for reconstruction.
"""

import argparse
import json
import os
from collections import Counter
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ═════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Action labels will be built dynamically from dataset
# These are set globally after loading data
ACTION_LABELS = []
ACTION_TO_IDX = {}
IDX_TO_ACTION = {}
NUM_ACTIONS = 0
STATE_ACTION_KEYS = []
STATE_DIM = 3  # x, y, z + yaw + pitch + action_counts

# ImageNet normalisation expected by torchvision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ResNet-18 produces a 512-d feature vector after avgpool
RESNET_FEAT_DIM = 512

METRICS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "metrics", "agent_metrics"
)

# ═════════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════════

class ActionDataset(Dataset):
    """
    Loads examples from a JSONL file produced by dataset.py.

    Each example contains:
      - image  : path to a screenshot
      - state  : {x, y, z, action_counts: {action: count, …}}
      - action : target label string
    """

    def __init__(self, entries, image_size=224, root_dir=".", augment=False):
        self.entries = entries
        self.root_dir = root_dir

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _state_vector(state):
        """Build a fixed-length float tensor from the state dict."""
        vec = [
            state.get("x", 0.0),
            state.get("y", 0.0),
            state.get("z", 0.0),
            state.get("yaw", 0.0),
            state.get("pitch", 0.0),
        ]
        counts = state.get("action_counts", {})
        for key in STATE_ACTION_KEYS:
            vec.append(float(counts.get(key, 0)))
        return torch.tensor(vec, dtype=torch.float32)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # ── image ──
        img_path = entry["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        # ── state ──
        state_vec = self._state_vector(entry["state"])

        # ── label ──
        label = ACTION_TO_IDX.get(entry["action"], 0)

        return image, state_vec, label


def load_jsonl(path):
    """Read a JSONL file and return a list of dicts."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_action_labels(entries):
    """Build action labels dynamically from dataset entries."""
    global ACTION_LABELS, ACTION_TO_IDX, IDX_TO_ACTION, NUM_ACTIONS, STATE_ACTION_KEYS, STATE_DIM
    
    # Extract unique actions and sort for reproducibility
    unique_actions = sorted(set(e["action"] for e in entries))
    
    ACTION_LABELS = unique_actions
    ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LABELS)}
    IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}
    NUM_ACTIONS = len(ACTION_LABELS)
    STATE_ACTION_KEYS = ACTION_LABELS
    STATE_DIM = 3 + 2 + len(STATE_ACTION_KEYS)  # x,y,z + yaw,pitch + action_counts
    
    print(f"Actions detected: {ACTION_LABELS}")
    print(f"Number of actions: {NUM_ACTIONS}")
    print(f"State dimension: {STATE_DIM}\n")


# ═════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════

class ActionPredictor(nn.Module):
    """
    Multimodal classifier:
      visual branch  → ResNet-18 (pretrained) → 512-d features
      state  branch  → raw state vector            (STATE_DIM)
      ──────────────────────────────────────────────────────────
      concat → MLP → softmax over NUM_ACTIONS

    Fine-tuning strategy (with ~4 k samples):
      - layer1, layer2, layer3: frozen  (generic low/mid-level features)
      - layer4 + head MLP     : trainable  (task-specific adaptation)
    """

    def __init__(self, state_dim=STATE_DIM, hidden_dim=256, num_actions=NUM_ACTIONS,
                 freeze_backbone=True):
        super().__init__()

        # ── visual backbone ──────────────────────────────────────────
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove original FC — we'll attach our own head
        self.backbone_features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.avgpool = backbone.avgpool  # outputs (B, 512, 1, 1)

        if freeze_backbone:
            # Freeze everything except layer4
            for name, param in self.backbone_features.named_parameters():
                # layer4 params have names like "7.0.conv1.weight" (index 7 in Sequential)
                if not name.startswith("7"):
                    param.requires_grad = False

        # ── classification head ──────────────────────────────────────
        fused_dim = RESNET_FEAT_DIM + state_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def _encode_image(self, x):
        """Run image through ResNet backbone → 512-d vector."""
        h = self.backbone_features(x)
        h = self.avgpool(h)
        h = h.view(h.size(0), -1)  # (B, 512)
        return h

    def forward(self, image, state):
        vis = self._encode_image(image)
        fused = torch.cat([vis, state], dim=1)
        return self.classifier(fused)


# ═════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels):
    """Inverse-frequency weights for balanced cross-entropy."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(NUM_ACTIONS)
    for idx in range(NUM_ACTIONS):
        count = counts.get(idx, 1)
        weights[idx] = total / (NUM_ACTIONS * count)
    return weights


def build_sampler(labels):
    """WeightedRandomSampler for balanced mini-batches."""
    counts = Counter(labels)
    class_weight = {c: len(labels) / n for c, n in counts.items()}
    sample_weights = [class_weight[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


# ═════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════

def train_model(
    jsonl_path,
    output_path="src/evaluation/models/action_predictor.pth",
    root_dir=".",
    image_size=224,
    hidden_dim=256,
    batch_size=32,
    epochs=30,
    lr=1e-4,
    val_split=0.15,
    seed=42,
):
    """Full training pipeline using ResNet-18 transfer learning."""
    print("/n" + "=" * 60)
    print("ACTION PREDICTOR — TRAINING (ResNet-18)")
    print("=" * 60)
    print(f"Device       : {DEVICE}")
    print(f"Dataset      : {jsonl_path}")
    print(f"Backbone     : ResNet-18 (ImageNet pretrained)")
    print(f"Image size   : {image_size}")
    print(f"Hidden dim   : {hidden_dim}")
    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {epochs}")
    print(f"LR           : {lr}")
    print("=" * 60 + "/n")

    start = time.time()
    # ── 1. Load data ─────────────────────────────────────────────────
    entries = load_jsonl(jsonl_path)
    
    # Build action labels dynamically from dataset
    build_action_labels(entries)
    
    labels = [ACTION_TO_IDX.get(e["action"], 0) for e in entries]
    
    # Check if stratify is possible (all classes need at least 2 samples)
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    use_stratify = min_count >= 2
    
    if not use_stratify:
        print(f"⚠️  Warning: Some classes have <2 samples. Stratification disabled.")
        print(f"    Class counts: {label_counts}\n")

    train_entries, val_entries, train_labels, val_labels = train_test_split(
        entries, labels, test_size=val_split, random_state=seed, 
        stratify=labels if use_stratify else None
    )

    print(f"Train: {len(train_entries)}  |  Val: {len(val_entries)}")
    print(f"Label distribution (train): {Counter(train_labels)}")
    print(f"Label distribution (val)  : {Counter(val_labels)}/n")

    train_ds = ActionDataset(train_entries, image_size=image_size, root_dir=root_dir, augment=True)
    val_ds   = ActionDataset(val_entries,   image_size=image_size, root_dir=root_dir, augment=False)

    sampler = build_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)

    # ── 2. Build model ───────────────────────────────────────────────
    model = ActionPredictor(
        state_dim=STATE_DIM, 
        hidden_dim=hidden_dim, 
        num_actions=NUM_ACTIONS
    ).to(DEVICE)

    # Show trainable vs frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total  "
          f"({100 * trainable / total:.1f}% trainable)/n")

    class_weights = compute_class_weights(train_labels).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 3. Train loop ────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # — train —
        model.train()
        running_loss, correct, total_samples = 0.0, 0, 0
        for images, states, targets in train_loader:
            images  = images.to(DEVICE)
            states  = states.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images, states)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss  += loss.item() * images.size(0)
            correct       += (logits.argmax(1) == targets).sum().item()
            total_samples += images.size(0)

        train_loss = running_loss / total_samples
        train_acc  = correct / total_samples

        # — validation —
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, states, targets in val_loader:
                images  = images.to(DEVICE)
                states  = states.to(DEVICE)
                targets = targets.to(DEVICE)

                logits = model(images, states)
                loss = criterion(logits, targets)

                val_loss    += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "image_size": image_size,
                "hidden_dim": hidden_dim,
                "state_dim": STATE_DIM,
                "num_actions": NUM_ACTIONS,
                "action_labels": ACTION_LABELS,
                "state_action_keys": STATE_ACTION_KEYS,
                "best_val_acc": best_val_acc,
            }, output_path)
            tag = "  ★ saved"

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}{tag}"
        )

    print(f"/nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {output_path}")
    print(f"Total training time: {(time.time() - start) / 60:.1f} minutes")
    return model, history


# ═════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════

def evaluate(
    jsonl_path,
    model_weights,
    root_dir=".",
    image_size=224,
    hidden_dim=256,
    batch_size=32,
):
    """Run evaluation on a full dataset and print classification report."""
    entries = load_jsonl(jsonl_path)
    
    # Build action labels dynamically from dataset
    build_action_labels(entries)

    dataset = ActionDataset(entries, image_size=image_size, root_dir=root_dir, augment=False)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ActionPredictor(
        state_dim=STATE_DIM, 
        hidden_dim=hidden_dim, 
        num_actions=NUM_ACTIONS
    ).to(DEVICE)
    ckpt  = torch.load(model_weights, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, states, targets in loader:
            images = images.to(DEVICE)
            states = states.to(DEVICE)
            logits = model(images, states)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(targets.tolist())

    print("/n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_targets, all_preds, target_names=ACTION_LABELS))

    return all_targets, all_preds


# ═════════════════════════════════════════════════════════════════════════
# PREDICTION (SINGLE EXAMPLE)
# ═════════════════════════════════════════════════════════════════════════

def predict_action(
    screenshot_path,
    state,
    model_weights,
    image_size=224,
    hidden_dim=256,
):
    """
    Predict the best action for a single game state.

    Parameters
    ----------
    screenshot_path : str   Path to a Minecraft screenshot.
    state           : dict  {"x": float, "y": float, "z": float,
                             "action_counts": {"mine": 5, …}}
    model_weights   : str   Path to the trained action predictor .pth

    Returns
    -------
    dict  {"action": str, "confidence": float, "probabilities": dict}
    """
    ckpt  = torch.load(model_weights, map_location=DEVICE, weights_only=False)
    
    # Load metadata from checkpoint
    saved_action_labels = ckpt.get("action_labels", [])
    saved_state_dim = ckpt.get("state_dim", 3)
    saved_num_actions = ckpt.get("num_actions", len(saved_action_labels))
    
    if saved_action_labels:
        # Temporarily set global action labels for prediction
        global ACTION_LABELS, ACTION_TO_IDX, IDX_TO_ACTION, NUM_ACTIONS, STATE_DIM, STATE_ACTION_KEYS
        ACTION_LABELS = saved_action_labels
        ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LABELS)}
        IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}
        NUM_ACTIONS = saved_num_actions
        STATE_DIM = saved_state_dim
        # Keep state encoding keys aligned with training-time feature layout.
        # Older checkpoints may not include explicit keys; fallback to action labels.
        STATE_ACTION_KEYS = ckpt.get("state_action_keys", saved_action_labels)
    
    model = ActionPredictor(
        state_dim=saved_state_dim, 
        hidden_dim=hidden_dim, 
        num_actions=saved_num_actions
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── prepare inputs ───────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image = Image.open(screenshot_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    state_tensor = ActionDataset._state_vector(state).unsqueeze(0).to(DEVICE)
    # Defensive compatibility: if checkpoint metadata and runtime keys diverge,
    # adapt vector size to the exact state_dim expected by the model.
    current_state_dim = state_tensor.shape[1]
    if current_state_dim < saved_state_dim:
        pad = torch.zeros((1, saved_state_dim - current_state_dim), device=DEVICE)
        state_tensor = torch.cat([state_tensor, pad], dim=1)
    elif current_state_dim > saved_state_dim:
        state_tensor = state_tensor[:, :saved_state_dim]

    # ── inference ────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(image_tensor, state_tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = probs.argmax().item()
    return {
        "action": IDX_TO_ACTION[pred_idx],
        "confidence": probs[pred_idx].item(),
        "probabilities": {IDX_TO_ACTION[i]: probs[i].item() for i in range(NUM_ACTIONS)},
    }


# ═════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_path="src/evaluation/plots/action_predictor_training.png"):
    """Plot loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion(targets, preds, save_path="src/evaluation/plots/action_predictor_confusion.png"):
    """Plot a confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=ACTION_LABELS, yticklabels=ACTION_LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Action Predictor")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Action predictor for HTN symbolic plan")
    sub = parser.add_subparsers(dest="command")

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the action predictor")
    p_train.add_argument("--jsonl",      default="data/train.jsonl", help="Path to JSONL dataset")
    p_train.add_argument("--output",     default="action_predictor.pth", help="Output model path")
    p_train.add_argument("--root_dir",   default=".", help="Root dir for relative image paths")
    p_train.add_argument("--image_size", type=int, default=224)
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--epochs",     type=int, default=30)
    p_train.add_argument("--lr",         type=float, default=1e-4)
    p_train.add_argument("--seed",       type=int, default=42)

    # ── eval ──────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--jsonl",      default="data/train.jsonl")
    p_eval.add_argument("--model",      default="action_predictor.pth")
    p_eval.add_argument("--root_dir",   default=".")
    p_eval.add_argument("--image_size", type=int, default=224)
    p_eval.add_argument("--hidden_dim", type=int, default=256)

    # ── predict ───────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Predict action for a single screenshot")
    p_pred.add_argument("--screenshot",  required=True, help="Path to screenshot")
    p_pred.add_argument("--state_json",  required=True, help="JSON string or file with state dict")
    p_pred.add_argument("--model",       default="action_predictor.pth")
    p_pred.add_argument("--image_size",  type=int, default=224)
    p_pred.add_argument("--hidden_dim",  type=int, default=256)

    # ── dataset ───────────────────────────────────────────────────────
    p_ds = sub.add_parser("dataset", help="(Re)build JSONL dataset from agent_metrics")
    p_ds.add_argument("--metrics_dir", default=METRICS_DIR)
    p_ds.add_argument("--output_jsonl", default="data/train.jsonl")
    p_ds.add_argument("--root_dir", default=".")

    args = parser.parse_args()

    if args.command == "train":
        model, history = train_model(
            jsonl_path=args.jsonl,
            output_path=args.output,
            root_dir=args.root_dir,
            image_size=args.image_size,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )
        plot_training_history(history)

        # Quick eval on full dataset
        targets, preds = evaluate(
            jsonl_path=args.jsonl,
            model_weights=args.output,
            root_dir=args.root_dir,
            image_size=args.image_size,
            hidden_dim=args.hidden_dim,
        )
        plot_confusion(targets, preds)

    elif args.command == "eval":
        targets, preds = evaluate(
            jsonl_path=args.jsonl,
            model_weights=args.model,
            root_dir=args.root_dir,
            image_size=args.image_size,
            hidden_dim=args.hidden_dim,
        )
        plot_confusion(targets, preds)

    elif args.command == "predict":
        # State can be a JSON file or inline JSON string
        if os.path.isfile(args.state_json):
            with open(args.state_json) as f:
                state = json.load(f)
        else:
            state = json.loads(args.state_json)

        result = predict_action(
            screenshot_path=args.screenshot,
            state=state,
            model_weights=args.model,
            image_size=args.image_size,
            hidden_dim=args.hidden_dim,
        )
        print(f"/nPredicted action : {result['action']}")
        print(f"Confidence       : {result['confidence']:.3f}")
        print("Probabilities    :")
        for action, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {action:8s} {prob:.3f} {bar}")

    elif args.command == "dataset":
        from dataset import build_dataset
        build_dataset(args.metrics_dir, args.output_jsonl, args.root_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

'''
# Entrenar
python src/evaluation/llm.py train

# Regenerar dataset desde métricas
python src/evaluation/llm.py dataset

# Evaluar
python src/evaluation/llm.py eval

# Predecir una acción (incluye x, y, z, yaw, pitch, action_counts)
python src/evaluation/llm.py predict \
  --screenshot "src/metrics/agent_metrics/A1_screenshots/control_2026-03-02T14-39-20-838Z.png" \
  --state_json '{"x":10.5,"y":64.0,"z":-5.3,"yaw":1.234,"pitch":-0.456,"action_counts":{"mine_pressed":3,"forward_pressed":5}}'

# Ejemplo mirando hacia el norte (yaw ≈ π/2)
python src/evaluation/llm.py predict \
  --screenshot "src/metrics/example.png" \
  --state_json '{"x":0,"y":70,"z":0,"yaw":1.57,"pitch":0,"action_counts":{}}'

# Ejemplo mirando hacia abajo para minar (pitch negativo)
python src/evaluation/llm.py predict \
  --screenshot "src/metrics/example2.png" \
  --state_json '{"x":15,"y":64,"z":-20,"yaw":0,"pitch":-0.8,"action_counts":{"mine_pressed":10}}'

# Ejemplo saltando (con historial de jumps)
python src/evaluation/llm.py predict \
  --screenshot "src/metrics/example3.png" \
  --state_json '{"x":-5,"y":68,"z":12,"yaw":3.14,"pitch":0.2,"action_counts":{"jump_pressed":7,"forward_pressed":12}}'
'''
