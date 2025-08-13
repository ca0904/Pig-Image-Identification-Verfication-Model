import copy
import os
import pickle
import random
from collections import Counter, defaultdict

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

SINGLE_COL_WIDTH = 6.5
SINGLE_COL_HEIGHT = 4.0
SQUARE_SIZE = 4.0
IMAGE_DIR = "pig-images/"
SAVE_DIR = "weights/"

mpl.rcParams.update(
    {
        "savefig.bbox": "tight",
        "savefig.format": "png",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
    }
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _save_fig(fig, filename):
    fig.tight_layout()
    fullpath = os.path.join(SAVE_DIR, filename)
    fig.savefig(fullpath, dpi=600)
    plt.close(fig)


def save_hist(
    data,
    filename,
    title,
    xlabel,
    figsize=(SINGLE_COL_WIDTH, SINGLE_COL_HEIGHT),
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=50, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    _save_fig(fig, filename)


def save_dual_hist(
    data1,
    data2,
    filename,
    title,
    labels,
    threshold=None,
    figsize=(SINGLE_COL_WIDTH, SINGLE_COL_HEIGHT),
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data1, bins=50, alpha=0.6, label=labels[0])
    ax.hist(data2, bins=50, alpha=0.6, label=labels[1])
    if threshold is not None:
        ax.axvline(
            threshold, color="k", linestyle="--", label=f"Threshold={threshold:.3f}"
        )
    ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("Similarity Score" if threshold is not None else "Pixels")
    ax.set_ylabel("Frequency")
    _save_fig(fig, filename)


def save_line_panel(history, filename, figsize=(SINGLE_COL_WIDTH, SINGLE_COL_HEIGHT)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax1.set_title("Loss History")
    ax2.set_title("Accuracy History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    _save_fig(fig, filename)


def save_square(plot_func, filename, title, xlabel, ylabel, annotate=None):
    fig, ax = plt.subplots(figsize=(SQUARE_SIZE, SQUARE_SIZE))
    plot_func(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if annotate:
        ax.text(
            0.05,
            0.95,
            annotate,
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
        )
    _save_fig(fig, filename)


def tune_blur_threshold(frames):
    lap_vars = [
        cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        for f in frames
    ]
    blur_threshold = np.percentile(lap_vars, 10)
    save_hist(lap_vars, "blur_threshold.png", "Laplacian Variance", "Variance")
    return blur_threshold


def tune_size_threshold(frames):
    widths = [f.shape[1] for f in frames]
    heights = [f.shape[0] for f in frames]
    width_threshold = np.percentile(widths, 10)
    height_threshold = np.percentile(heights, 10)
    save_dual_hist(
        widths,
        heights,
        "width_height_threshold.png",
        "Image Size Distribution",
        ["widths", "heights"],
    )
    return width_threshold, height_threshold


def filter_dataset(image_dir):
    images, dataset = [], []
    for dir in sorted(os.listdir(image_dir)):
        for img in sorted(os.listdir(os.path.join(image_dir, dir))):
            images.append(cv2.imread(os.path.join(image_dir, dir, img)))
            dataset.append(os.path.join(image_dir, dir, img))

    blur = tune_blur_threshold(images)
    width, height = tune_size_threshold(images)

    filtered_dataset = defaultdict(list)
    for image, path in zip(images, dataset):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < blur:
            continue

        h, w = image.shape[:2]
        if h < height or w < width:
            continue

        aspect_ratio = w / h
        if not (0.75 <= aspect_ratio <= 1.33):
            continue

        dir = os.path.dirname(path).split("/")[-1]
        filtered_dataset[dir].append(path)

    return filtered_dataset


def prepare_dataset_splits(image_dir, known_unknown_split=0.7):
    np.random.seed(42)
    pig_folders = sorted([f for f in os.listdir(image_dir)])
    num_known = int(known_unknown_split * len(pig_folders))
    np.random.shuffle(pig_folders)
    known_pigs = pig_folders[:num_known]
    unknown_pigs = pig_folders[num_known:]
    return sorted(known_pigs), sorted(unknown_pigs)


def prepare_known_splits(
    filtered_dataset, known_classes, known_split=[0.7, 0.15, 0.15]
):
    np.random.seed(42)
    train_samples, known_val_samples, known_test_samples = [], [], []
    class_to_idx = {pig: i for i, pig in enumerate(known_classes)}
    for pig in known_classes:
        images = filtered_dataset[pig]
        np.random.shuffle(images)

        n_images = len(images)
        n_train = int(known_split[0] * n_images)
        adjusted_ratio = known_split[1] / (1 - known_split[0])
        n_val = int(adjusted_ratio * (n_images - n_train))

        train_images = images[:n_train]
        val_images = images[n_train : n_train + n_val]
        test_images = images[n_train + n_val :]

        train_samples += [(img, class_to_idx[pig], pig) for img in train_images]
        known_val_samples += [(img, class_to_idx[pig], pig) for img in val_images]
        known_test_samples += [(img, class_to_idx[pig], pig) for img in test_images]

    return train_samples, known_val_samples, known_test_samples


def prepare_unknown_splits(filtered_dataset, unknown_classes, unknown_split=[0.5, 0.5]):
    np.random.seed(42)
    unknown_val_samples, unknown_test_samples = [], []
    for pig in unknown_classes:
        images = filtered_dataset[pig]
        np.random.shuffle(images)

        n_images = len(images)
        n_val = int(unknown_split[0] * n_images)

        val_images = images[:n_val]
        test_images = images[n_val:]

        unknown_val_samples += [(img, -1, pig) for img in val_images]
        unknown_test_samples += [(img, -1, pig) for img in test_images]

    return unknown_val_samples, unknown_test_samples


class PigDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label, class_name if label != -1 else "unknown", img_path


class PairDataset(Dataset):
    def __init__(self, samples, num_pairs, transform=None):
        self.samples = samples
        self.num_pairs = num_pairs
        self.transform = transform
        self.class_name_to_indices = self._group_by_class_name()
        self.pairs = self._create_pairs()

    def _group_by_class_name(self):
        class_name_to_indices = defaultdict(list)
        for idx, (_, _, class_name) in enumerate(self.samples):
            class_name_to_indices[class_name].append(idx)
        return class_name_to_indices

    def _create_pairs(self):
        pairs = []
        seen = set()
        class_names = list(self.class_name_to_indices.keys())
        num_pos = self.num_pairs // 2
        num_neg = self.num_pairs - num_pos

        generated_pos = 0
        while generated_pos < num_pos:
            class_name = np.random.choice(class_names)
            indices = self.class_name_to_indices[class_name]
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
            key = tuple(sorted((idx1, idx2)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key, 1))
            generated_pos += 1

        generated_neg = 0
        while generated_neg < num_neg:
            class_name1, class_name2 = np.random.choice(class_names, 2, replace=False)
            idx1 = np.random.choice(self.class_name_to_indices[class_name1])
            idx2 = np.random.choice(self.class_name_to_indices[class_name2])
            key = tuple(sorted((idx1, idx2)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key, 0))
            generated_neg += 1

        np.random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (idx1, idx2), pair_label = self.pairs[idx]
        img1_path, _, _ = self.samples[idx1]
        img2_path, _, _ = self.samples[idx2]
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, pair_label, img1_path, img2_path


def get_transforms(phase):
    if phase == "train":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    return transform


def get_dataloader(samples, phase, num_pairs=None):
    transform = get_transforms(phase)
    if num_pairs is not None:
        dataset = PairDataset(samples, num_pairs, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    elif phase == "train":
        labels = [label for _, label, _ in samples]
        class_counts = Counter(labels)
        num_samples = len(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True
        )
        dataset = PigDataset(samples, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataset = PigDataset(samples, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    return dataloader


def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def plot_data_augmentation(
    filtered_dataset, transform, save_path="data_augmentation.png"
):
    classes = random.sample(list(filtered_dataset.keys()), 3)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i, cls in enumerate(classes):
        img_path = random.choice(filtered_dataset[cls])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        for j in range(3):
            augmented = transform(img)
            augmented_img = unnormalize(augmented)
            axes[i, j + 1].imshow(augmented_img)
            axes[i, j + 1].set_title(f"Aug {j + 1}")
            axes[i, j + 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=600)
    plt.close()


class ArcFaceHead(nn.Module):
    def __init__(self, feature_dim, num_classes, s=30, m=0.5):
        super().__init__()
        self.scale, self.margin = s, m
        self.weight = nn.Parameter(torch.Tensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        x = F.normalize(features)
        w = F.normalize(self.weight)
        cos_theta = F.linear(x, w)
        theta = torch.acos(torch.clamp(cos_theta, 1e-7 - 1.0, 1.0 - 1e-7))
        target_cos = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = cos_theta * (1 - one_hot) + target_cos * one_hot
        return logits * self.scale


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_c=0.003):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        loss = F.mse_loss(features, centers_batch)
        return self.lambda_c * loss


class EfficientNetWithHeads(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_classes,
        dropout=0.1,
        s=30,
        m=0.5,
        lambda_c=0.003,
    ):
        super().__init__()
        backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, global_pool="avg"
        )
        backbone.load_state_dict(
            torch.load(
                os.path.join(SAVE_DIR, "base_model.pt"),
                map_location="cpu",
                weights_only=True,
            ),
        )
        blocks = list(backbone.blocks)
        freeze_count = int(len(blocks) * 0.75)
        for block in blocks[:freeze_count]:
            for param in block.parameters():
                param.requires_grad = False
        self.backbone = backbone
        in_features = backbone.num_features

        layers = []
        prev_dim = in_features
        for _ in range(3):
            layers.append(nn.Linear(prev_dim, 128, bias=False))
            layers.append(nn.BatchNorm1d(128))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = 128
        layers.append(nn.Linear(prev_dim, feature_dim))

        self.feature_block = nn.Sequential(*layers)
        self.arcface = ArcFaceHead(feature_dim, num_classes, s, m)
        self.center_loss = CenterLoss(num_classes, feature_dim, lambda_c)

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        feat = self.feature_block(feat)
        feat = F.normalize(feat)
        if labels is not None:
            logits = self.arcface(feat, labels)
            c_loss = self.center_loss(feat, labels)
            return logits, feat, c_loss
        return feat


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def __call__(self, current_score):
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train(model, loader, optimizer, optimizer_center, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, _, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        logits, _, c_loss = model(imgs, labels)
        ce = F.cross_entropy(logits, labels)
        loss = ce + c_loss
        loss.backward()
        optimizer.step()
        optimizer_center.step()
        total_loss += loss.item() * imgs.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader.dataset), correct / total


def validate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds = []
    with torch.no_grad():
        for imgs, labels, _, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _, c_loss = model(imgs, labels)
            ce = F.cross_entropy(logits, labels)
            loss = ce + c_loss
            total_loss += loss.item() * imgs.size(0)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader.dataset), correct / total


def train_model(train_loader, val_loader, num_known, device, best, save=False):
    model = EfficientNetWithHeads(
        best["feat_dim"],
        num_known,
        best["dropout"],
        best["s"],
        best["m"],
        best["lambda_c"],
    ).to(device)

    params = (
        list(model.backbone.parameters())
        + list(model.feature_block.parameters())
        + list(model.arcface.parameters())
    )
    opt = AdamW(params, lr=best["lr"], weight_decay=best["wd"])
    opt_center = SGD(
        model.center_loss.parameters(),
        lr=best["lr_center"],
        momentum=best["momentum_center"],
        weight_decay=best["wd_center"],
    )
    sched = CosineAnnealingWarmRestarts(opt, T_0=best["T_0"], eta_min=best["eta_min"])
    sched_center = StepLR(
        opt_center, step_size=best["step_size_center"], gamma=best["gamma_center"]
    )

    best_acc = 0.0
    early_stopping = EarlyStopping()
    best_model = copy.deepcopy(model.state_dict())
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(100):
        train_loss, train_acc = train(model, train_loader, opt, opt_center, device)
        val_loss, val_acc = validate(model, val_loader, device)
        sched.step()
        sched_center.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

        if early_stopping(val_acc):
            break

    model.load_state_dict(best_model)

    if not save:
        return model

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
    df = pd.DataFrame(history)
    df.rename(
        columns={
            "epoch": "Epoch",
            "train_loss": "Train Loss",
            "train_acc": "Train Accuracy",
            "val_loss": "Validation Loss",
            "val_acc": "Validation Accuracy",
        },
        inplace=True,
    )
    df.index = range(1, len(df) + 1)
    df.index.name = "Epoch"
    df.to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)
    save_line_panel(history, "training_history.png")
    return model


def extract_embeddings(model, device, loader):
    model.eval()
    embeddings = defaultdict(list)
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            feats = model(images)
            feats = F.normalize(feats)
            for feat, label in zip(feats, labels):
                embeddings[label.item()].append(feat)
    for label, feats in embeddings.items():
        embeddings[label] = torch.stack(feats, dim=0)
    embeddings = dict(sorted(embeddings.items()))
    return embeddings


def plot_tsne(gallery, label_to_class, save_path="tsne_gallery.png"):
    all_embeddings = []
    all_labels = []

    for label, feats in gallery.items():
        all_embeddings.append(feats.cpu().numpy())
        all_labels.extend([label] * feats.size(0))

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    tsne = TSNE(n_components=2, perplexity=5, init="random", random_state=42)
    tsne_result = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(gallery))
    for label in sorted(gallery.keys()):
        idx = all_labels == label
        plt.scatter(
            tsne_result[idx, 0],
            tsne_result[idx, 1],
            label=label_to_class[label],
            s=60,
            alpha=0.7,
            color=palette[label],
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE Plot of Prototype Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, save_path), dpi=600)
    plt.close()


def build_prototype_gallery(
    embeddings, num_prototypes=3, known_classes=None, save=False
):
    gallery = {}
    for label, feats in embeddings.items():
        feats = feats.cpu().numpy()
        n_samples, _ = feats.shape
        k = min(num_prototypes, n_samples)
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        km.fit(feats)
        centers = km.cluster_centers_
        prototypes = torch.from_numpy(centers)
        gallery[label] = F.normalize(prototypes)

    if not save:
        return gallery

    label_to_class = {i: known_classes[i] for i in range(len(known_classes))}
    plot_tsne(gallery, label_to_class)
    with open(os.path.join(SAVE_DIR, "feature_gallery.pkl"), "wb") as f:
        pickle.dump(
            {
                "gallery": gallery,
                "label_to_class": label_to_class,
            },
            f,
        )
    return gallery


def compute_scores(model, gallery, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, _, _, _ in loader:
            imgs = imgs.to(device)
            feats = F.normalize(model(imgs.to(device)))
            feats = feats.cpu().numpy()
            for f in feats:
                score = max([(protos @ f).max().item() for protos in gallery.values()])
                scores.append(score)
    return np.array(scores)


def tune_osr_threshold(model, gallery, known_loader, unknown_loader, device, plot=True):
    known_scores = compute_scores(model, gallery, known_loader, device)
    unknown_scores = compute_scores(model, gallery, unknown_loader, device)

    y_true = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])
    y_scores = np.concatenate([known_scores, unknown_scores])

    thresholds = np.unique(y_scores)
    f1s = np.array(
        [f1_score(y_true, (y_scores >= thr).astype(int)) for thr in thresholds]
    )

    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    if not plot:
        return best_thresh

    def plot_thres(ax):
        ax.plot(thresholds, f1s, linewidth=2)

    save_square(
        plot_thres,
        "osr_threshold.png",
        "Open-Set Recognition: F1 vs Threshold",
        "Threshold",
        "F1 Score",
        annotate=f"Best={best_f1:.3f} @ Thr={best_thresh:.3f}",
    )
    return best_thresh


def compute_pair_scores(model, loader, device):
    model.eval()
    sims, labels = [], []
    with torch.no_grad():
        for img1, img2, lbl, _, _ in loader:
            e1 = F.normalize(model(img1.to(device)))
            e2 = F.normalize(model(img2.to(device)))
            sim = (e1 * e2).sum(dim=1).cpu().numpy()
            sims.append(sim)
            labels.append(lbl.numpy())
    return np.concatenate(sims), np.concatenate(labels)


def tune_pair_threshold(model, pair_loader, device, plot=True):
    sims, labels = compute_pair_scores(model, pair_loader, device)

    thresholds = np.unique(sims)
    f1s = np.array([f1_score(labels, (sims >= thr).astype(int)) for thr in thresholds])

    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    best_thresh = float(best_thresh)

    if not plot:
        return best_thresh

    def plot_thres(ax):
        ax.plot(thresholds, f1s, linewidth=2)

    save_square(
        plot_thres,
        "verif_threshold.png",
        "Pair Verification: F1 vs Threshold",
        "Similarity Threshold",
        "F1 Score",
        annotate=f"Best={best_f1:.3f} @ Thr={best_thresh:.3f}",
    )
    return best_thresh


def evaluate_osr_identification(
    model, gallery, known_loader, unknown_loader, device, threshold, plot=True
):
    model.eval()
    records = []
    label_to_class = {}

    def process_batch(imgs):
        imgs = imgs.to(device)
        feats = F.normalize(model(imgs), dim=1)
        sims = (
            torch.stack(
                [
                    (feats @ proto.to(device).T).max(dim=1)[0]
                    for proto in gallery.values()
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        max_sims = sims.max(axis=1)
        argmax_ids = sims.argmax(axis=1)
        return max_sims, argmax_ids

    for imgs, labels, class_names, paths in known_loader:
        sim, argm = process_batch(imgs)
        for s, a, true, p in zip(sim, argm, labels.numpy(), paths):
            records.append(
                {"score": s, "argmax": int(a), "true": int(true), "img_path": p}
            )
        for label, class_name in zip(labels.numpy(), class_names):
            if label not in label_to_class:
                label_to_class[label] = class_name

    label_to_class = dict(sorted(label_to_class.items()))
    label_to_class[-1] = "unknown"

    for imgs, _, _, paths in unknown_loader:
        sim, argm = process_batch(imgs)
        for s, a, p in zip(sim, argm, paths):
            records.append({"score": s, "argmax": int(a), "true": -1, "img_path": p})

    scores = np.array([r["score"] for r in records])
    argmaxes = np.array([r["argmax"] for r in records])
    trues = np.array([r["true"] for r in records])
    paths = [r["img_path"] for r in records]

    preds = np.where(scores >= threshold, argmaxes, -1)

    acc_c = accuracy_score(trues[trues != -1], preds[trues != -1])
    acc_oc = accuracy_score(trues, preds)
    prec_macro = precision_score(trues, preds, average="macro", zero_division=0)
    rec_macro = recall_score(trues, preds, average="macro", zero_division=0)
    prec_micro = precision_score(trues, preds, average="micro", zero_division=0)
    rec_micro = recall_score(trues, preds, average="micro", zero_division=0)
    f1_macro = f1_score(trues, preds, average="macro", zero_division=0)
    f1_micro = f1_score(trues, preds, average="micro", zero_division=0)

    labels_list = list(gallery.keys()) + [-1]
    p, r, f1, sup = precision_recall_fscore_support(
        trues, preds, labels=labels_list, zero_division=0
    )
    per_cls = {}
    for i, cls in enumerate(labels_list):
        per_cls[label_to_class[cls]] = {
            "accuracy": (
                (preds[trues == cls] == trues[trues == cls]).mean() if sup[i] > 0 else 0
            ),
            "precision": p[i],
            "recall": r[i],
            "f1": f1[i],
            "support": sup[i],
            "misclassified": (preds[trues == cls] != trues[trues == cls]).sum(),
        }

    y_det_true = (trues != -1).astype(int)
    y_det_scores = scores
    osr_auroc = roc_auc_score(y_det_true, y_det_scores)
    prec_curve, rec_curve, _ = precision_recall_curve(y_det_true, scores)
    osr_aupr = auc(rec_curve, prec_curve)
    y_det_pred = (y_det_scores >= threshold).astype(int)
    osr_acc = accuracy_score(y_det_true, y_det_pred)
    osr_prec = precision_score(y_det_true, y_det_pred, zero_division=0)
    osr_rec = recall_score(y_det_true, y_det_pred, zero_division=0)
    osr_f1 = f1_score(y_det_true, y_det_pred, zero_division=0)
    far = ((y_det_true == 0) & (y_det_pred == 1)).sum() / (y_det_true == 0).sum()
    ccr = accuracy_score(trues, preds)
    oscr = (1 - far) * ccr

    metrics = {
        "identification_accuracy_closed_set": acc_c,
        "identification_accuracy_open_closed_set": acc_oc,
        "identification_precision_macro_avg": prec_macro,
        "identification_recall_macro_avg": rec_macro,
        "identification_f1_macro_avg": f1_macro,
        "identification_precision_micro_avg": prec_micro,
        "identification_recall_micro_avg": rec_micro,
        "identification_f1_micro_avg": f1_micro,
        "identification_per_class_metrics": per_cls,
        "open_set_accuracy": osr_acc,
        "open_set_precision": osr_prec,
        "open_set_recall": osr_rec,
        "open_set_f1_score": osr_f1,
        "open_set_auroc": osr_auroc,
        "open_set_aupr": osr_aupr,
        "open_set_false_accept_rate": far,
        "open_set_correct_classification_rate": ccr,
        "open_set_classification_rate": oscr,
    }

    miscls = []
    for t, p, s, path in zip(trues, preds, scores, paths):
        if t != p:
            miscls.append(
                {
                    "img_path": path,
                    "true": label_to_class[t],
                    "pred": label_to_class[p],
                    "score": float(s),
                }
            )

    if not plot:
        return metrics, miscls

    cm = confusion_matrix(trues, preds, labels=labels_list)
    with plt.rc_context(rc=None):
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            cm,
            annot=False,
            fmt="g",
            cmap="Blues",
            cbar=True,
            square=True,
            xticklabels=label_to_class.values(),
            yticklabels=label_to_class.values(),
        )
        ax.tick_params(axis="x", rotation=90, labelsize=12)
        ax.tick_params(axis="y", rotation=0, labelsize=12)
        ax.set_xlabel("Predicted", fontsize=16)
        ax.set_ylabel("True", fontsize=16)
        ax.set_title("Identification Confusion Matrix", fontsize=18)
        plt.grid(False)
        plt.tight_layout()
        fullpath = os.path.join(SAVE_DIR, "id_confusion_matrix.png")
        plt.savefig(fullpath, dpi=600)
        plt.close()

    fpr, tpr, _ = roc_curve(y_det_true, y_det_scores)

    def plot_osr_roc(ax):
        ax.plot(fpr, tpr, linewidth=2, label="ROC curve")
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.legend(loc="lower right")

    def plot_osr_pr(ax):
        ax.plot(rec_curve, prec_curve, linewidth=2)

    save_square(
        plot_osr_roc,
        "osr_roc.png",
        title=f"OSR ROC (AUC={osr_auroc:.3f})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        annotate=f"AUC={osr_auroc:.3f}",
    )
    save_square(
        plot_osr_pr,
        "osr_pr.png",
        title=f"OSR PR (AUC={osr_aupr:.3f})",
        xlabel="Recall",
        ylabel="Precision",
    )
    save_dual_hist(
        y_det_scores[y_det_true == 1],
        y_det_scores[y_det_true == 0],
        "osr_similarity_score.png",
        "OSR Similarity Distributions",
        ["Known", "Unknown"],
        threshold,
    )

    return metrics, miscls


def evaluate_verification(model, pair_loader, device, threshold, plot=True):
    model.eval()
    sims = []
    labels = []
    paths1, paths2 = [], []

    with torch.no_grad():
        for img1, img2, lbl, p1, p2 in pair_loader:
            e1 = F.normalize(model(img1.to(device)), dim=1)
            e2 = F.normalize(model(img2.to(device)), dim=1)
            sim = (e1 * e2).sum(dim=1).cpu().numpy()
            sims.append(sim)
            labels.append(lbl.numpy())
            paths1.extend(p1)
            paths2.extend(p2)

    sims = np.concatenate(sims)
    labels = np.concatenate(labels)

    preds = (sims >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auroc = roc_auc_score(labels, sims)
    prec_curve, rec_curve, _ = precision_recall_curve(labels, sims)
    aupr = auc(rec_curve, prec_curve)
    avg_prec = average_precision_score(labels, sims)
    fpr, tpr, _ = roc_curve(labels, sims)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    far = ((labels == 0) & (preds == 1)).sum() / (labels == 0).sum()
    frr = ((labels == 1) & (preds == 0)).sum() / (labels == 1).sum()

    false_rejects = []
    false_accepts = []
    for s, y, yhat, p1, p2 in zip(sims, labels, preds, paths1, paths2):
        if y == 1 and yhat == 0:
            false_rejects.append({"path1": p1, "path2": p2, "score": float(s)})
        elif y == 0 and yhat == 1:
            false_accepts.append({"path1": p1, "path2": p2, "score": float(s)})

    report = {"false_rejects": false_rejects, "false_accepts": false_accepts}

    metrics = {
        "verification_accuracy": acc,
        "verification_precision": prec,
        "verification_recall": rec,
        "verification_f1_score": f1,
        "verification_auroc": auroc,
        "verification_aupr": aupr,
        "verification_average_precision": avg_prec,
        "verification_equal_error_rate": eer,
        "verification_false_acceptance_rate": far,
        "verification_false_rejection_rate": frr,
    }

    if not plot:
        return metrics, report

    cm = confusion_matrix(labels, preds)

    def plot_verif_cm(ax):
        ax.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Impostor", "Genuine"], rotation=0)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Impostor", "Genuine"], rotation=90)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

    def plot_verif_roc(ax):
        ax.plot(fpr, tpr, linewidth=2, label="ROC curve")
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.legend(loc="lower right")

    def plot_verif_pr(ax):
        ax.plot(rec_curve, prec_curve, linewidth=2)

    def plot_det(ax):
        ax.plot(fpr, fnr, linewidth=2, label="DET Curve")
        ax.plot([0, 1], [0, 1], "k--", label="EER Line")
        ax.scatter(fpr[eer_idx], fnr[eer_idx], color="red", label=f"EER={eer:.3f}")
        ax.legend(loc="lower right")

    save_square(
        plot_verif_cm,
        "verif_confusion_matrix.png",
        title="Verification Confusion Matrix",
        xlabel="Predicted",
        ylabel="True",
    )
    save_square(
        plot_verif_roc,
        "verif_roc.png",
        title=f"Verification ROC (AUC={auroc:.3f})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        annotate=f"AUC={auroc:.3f}",
    )
    save_square(
        plot_verif_pr,
        "verif_pr.png",
        title=f"Verification PR (AUC={aupr:.3f})",
        xlabel="Recall",
        ylabel="Precision",
    )
    save_square(
        plot_det,
        "verif_eer_curve.png",
        title="DET Curve",
        xlabel="False Positive Rate",
        ylabel="False Negative Rate",
    )
    save_dual_hist(
        sims[labels == 1],
        sims[labels == 0],
        "verif_similarity_score.png",
        "Verification Similarity Distributions",
        ["Genuine", "Impostor"],
        threshold,
    )

    return metrics, report


def cross_validate(
    filtered_dataset,
    known_classes,
):
    known_samples, labels = [], []
    class_to_idx = {pig: i for i, pig in enumerate(known_classes)}
    for pig in known_classes:
        images = filtered_dataset[pig]
        for img in images:
            known_samples.append((img, class_to_idx[pig], pig))
            labels.append(class_to_idx[pig])

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(known_samples, labels)
    ):
        outer_train = [known_samples[i] for i in outer_train_idx]
        outer_train_labels = [labels[i] for i in outer_train_idx]
        outer_test = [known_samples[i] for i in outer_test_idx]

        with open(os.path.join(SAVE_DIR, f"outer_{outer_idx + 1}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "outer_train": outer_train,
                    "outer_train_labels": outer_train_labels,
                    "outer_test": outer_test,
                },
                f,
            )
