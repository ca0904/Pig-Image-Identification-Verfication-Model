import json
import os
import pickle
import zipfile

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from PIL import Image

from utils import (
    SAVE_DIR,
    prepare_known_splits,
    get_transforms,
    get_dataloader,
    plot_data_augmentation,
    train_model,
    build_prototype_gallery,
    extract_embeddings,
    tune_osr_threshold,
    tune_pair_threshold,
    evaluate_osr_identification,
    evaluate_verification,
)

device = "cuda"

with open(os.path.join(SAVE_DIR, "dataset.pkl"), "rb") as f:
    data = pickle.load(f)

filtered_dataset = data["filtered_dataset"]
known_classes = data["known_classes"]
num_known = data["num_known"]
unknown_val_samples = data["unknown_val"]
unknown_test_samples = data["unknown_test"]

trials = []
outer_metrics = []
best_params_list = []
for fold in range(1, 6):
    with open(os.path.join(SAVE_DIR, f"study_{fold}.pkl"), "rb") as f:
        study = pickle.load(f)

    trial_df = study.trials_dataframe()
    trial_df["fold"] = f"Fold {fold}"
    trials.append(trial_df)
    best_params_list.append((study.best_trial.params, study.best_trial.value))
    best = study.best_trial.params

    with open(os.path.join(SAVE_DIR, f"outer_{fold}.pkl"), "rb") as f:
        data = pickle.load(f)

    outer_train = data["outer_train"]
    outer_test = data["outer_train"]

    train_loader = get_dataloader(outer_train, "train")
    known_test_loader = get_dataloader(outer_test, "test")
    unknown_test_loader = get_dataloader(unknown_test_samples, "test")
    pair_test_loader = get_dataloader(
        outer_test + unknown_test_samples, "test", num_pairs=20000
    )

    model = train_model(train_loader, known_test_loader, num_known, device, best)
    gallery = build_prototype_gallery(
        extract_embeddings(model, device, train_loader), num_prototypes=3
    )

    osr_threshold = tune_osr_threshold(
        model, gallery, known_test_loader, unknown_test_loader, device, plot=False
    )
    verif_threshold = tune_pair_threshold(model, pair_test_loader, device, plot=False)

    metrics_osr_id, _ = evaluate_osr_identification(
        model,
        gallery,
        known_test_loader,
        unknown_test_loader,
        device,
        osr_threshold,
        plot=False,
    )
    metrics_verif, _ = evaluate_verification(
        model, pair_test_loader, device, verif_threshold, plot=False
    )

    metrics = {}
    for key, value in metrics_osr_id.items():
        if key == "identification_per_class_metrics":
            metrics["osr_decision_threshold"] = osr_threshold
            continue
        metrics[key] = value
    metrics["verification_decision_threshold"] = verif_threshold
    for key, value in metrics_verif.items():
        metrics[key] = value

    outer_metrics.append(metrics)

df = pd.DataFrame(outer_metrics)
rename_dict = {
    "osr_decision_threshold": "Open-Set Decision Threshold",
    "verification_decision_threshold": "Verification Decision Threshold",
    "identification_accuracy_closed_set": "Identification Accuracy Closed Set",
    "identification_accuracy_open_closed_set": "Identification Accuracy Open Closed Set",
    "identification_precision_macro_avg": "Identification Precision Macro Avg",
    "identification_recall_macro_avg": "Identification Recall Macro Avg",
    "identification_f1_macro_avg": "Identification F1 Macro Avg",
    "identification_precision_micro_avg": "Identification Precision Micro Avg",
    "identification_recall_micro_avg": "Identification Recall Micro Avg",
    "identification_f1_micro_avg": "Identification F1 Micro Avg",
    "open_set_accuracy": "Open-Set Accuracy",
    "open_set_precision": "Open-Set Precision",
    "open_set_recall": "Open-Set Recall",
    "open_set_f1_score": "Open-Set F1 Score",
    "open_set_auroc": "Open-Set AUROC",
    "open_set_aupr": "Open-Set AUPR",
    "open_set_false_accept_rate": "Open-Set False Accept Rate",
    "open_set_correct_classification_rate": "Open-Set Correct Classification Rate",
    "open_set_classification_rate": "Open-Set Classification Rate",
    "verification_accuracy": "Verification Accuracy",
    "verification_precision": "Verification Precision",
    "verification_recall": "Verification Recall",
    "verification_f1_score": "Verification F1 Score",
    "verification_auroc": "Verification AUROC",
    "verification_aupr": "Verification AUPR",
    "verification_average_precision": "Verification Average Precision",
    "verification_equal_error_rate": "Verification Equal Error Rate",
    "verification_false_acceptance_rate": "Verification False Acceptance Rate",
    "verification_false_rejection_rate": "Verification False Rejection Rate",
}
df.rename(
    columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True
)
df.index = [f"Fold {i}" for i in range(1, len(df) + 1)]
df.index.name = "Cross-Validation Fold"
df.loc["Mean"] = df.mean(numeric_only=True)
df.loc["Std"] = df.std(numeric_only=True)
df.to_csv(os.path.join(SAVE_DIR, "cv_results.csv"))

rename_dict = {
    "fold": "Cross-Validation Fold",
    "number": "Trial Number",
    "value": "Score",
    "params_feat_dim": "Feature Dimensions",
    "params_dropout": "Dropout Rate",
    "params_s": "ArcFace Scaling Factor",
    "params_m": "ArcFace Angular Margin Penalty",
    "params_lambda_c": "Lambda Center",
    "params_lr": "Learning Rate",
    "params_wd": "Weight Decay",
    "params_lr_center": "Learning Rate Center",
    "params_wd_center": "Weight Decay Center",
    "params_momentum_center": "Momentum Center",
    "params_T_0": "Scheduler T0",
    "params_eta_min": "Scheduler Eta Min",
    "params_step_size_center": "Scheduler Center Step Size",
    "params_gamma_center": "Scheduler Center Gamma",
    "state": "State",
}
df = pd.concat(trials, ignore_index=True)
df.rename(columns=rename_dict, inplace=True)
df = df[[value for value in rename_dict.values() if value in df.columns]]
df.to_csv(os.path.join(SAVE_DIR, "cv_params_trials.csv"), index=False)

rename_dict = {
    key[7:] if key.startswith("params_") else key: value
    for key, value in rename_dict.items()
}

df = pd.DataFrame([value[0] for value in best_params_list])
df.rename(columns=rename_dict, inplace=True)
df.index = [f"CV Fold {i}" for i in range(1, len(df) + 1)]
df.index.name = "Cross-Validation Fold"
df.insert(0, "Score", [value[1] for value in best_params_list])
df.to_csv(os.path.join(SAVE_DIR, "best_params.csv"))

best_params = max(best_params_list, key=lambda x: x[1])[0]

train_samples, known_val_samples, known_test_samples = prepare_known_splits(
    filtered_dataset, known_classes
)

train_loader = get_dataloader(train_samples, "train")
known_val_loader = get_dataloader(known_val_samples, "val")
known_test_loader = get_dataloader(known_test_samples, "test")
unknown_val_loader = get_dataloader(unknown_val_samples, "val")
unknown_test_loader = get_dataloader(unknown_test_samples, "test")
pair_val_loader = get_dataloader(
    known_val_samples + unknown_val_samples, "val", num_pairs=10000
)
pair_test_loader = get_dataloader(
    known_test_samples + unknown_test_samples, "test", num_pairs=20000
)

plot_data_augmentation(filtered_dataset, get_transforms("train"))

model = train_model(
    train_loader, known_val_loader, num_known, device, best_params, save=True
)
gallery = build_prototype_gallery(
    extract_embeddings(model, device, train_loader),
    num_prototypes=3,
    known_classes=known_classes,
    save=True,
)

osr_threshold = tune_osr_threshold(
    model, gallery, known_val_loader, unknown_val_loader, device
)
verif_threshold = tune_pair_threshold(model, pair_val_loader, device)

metrics_osr_id, miscls = evaluate_osr_identification(
    model, gallery, known_test_loader, unknown_test_loader, device, osr_threshold
)
metrics_verif, report = evaluate_verification(
    model, pair_test_loader, device, verif_threshold
)

per_class_df = pd.DataFrame.from_dict(
    metrics_osr_id["identification_per_class_metrics"], orient="index"
)
per_class_df.reset_index(inplace=True)
per_class_df.rename(columns={"index": "class_name"}, inplace=True)
per_class_df.rename(
    columns={
        "class_name": "Class Name",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "support": "Support",
        "misclassified": "Misclassified Count",
    },
    inplace=True,
)
per_class_df.to_csv(os.path.join(SAVE_DIR, "id_per_class_metrics.csv"), index=False)

miscls_df = pd.DataFrame(miscls)
miscls_df.rename(
    columns={
        "img_path": "Image Path",
        "true": "True Label",
        "pred": "Predicted Label",
        "score": "Prediction Confidence",
    },
    inplace=True,
)
miscls_df.to_csv(os.path.join(SAVE_DIR, "misclassified_samples.csv"), index=False)

n = min(9, len(miscls_df))
rows = (n + 2) // 3
plt.figure(figsize=(10, 10))
for i in range(n):
    row = miscls_df.iloc[i]
    img = Image.open(row["Image Path"])
    ax = plt.subplot(rows, 3, i + 1)
    ax.imshow(img)
    ax.axis("off")
    title_str = (
        f"Pred: {row['Predicted Label']}\n"
        f"True: {row['True Label']}\n"
        f"Conf: {row['Prediction Confidence']:.2f}"
    )
    ax.set_title(title_str, fontsize=10, pad=5)

plt.suptitle("Misclassified Samples", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "misclassified_samples.png"), dpi=600)
plt.close()

df_false_rejects = pd.DataFrame(report["false_rejects"])
df_false_rejects.rename(
    columns={
        "path1": "Image 1: Path",
        "path2": "Image 2: Path",
        "score": "Similarity Score",
    },
    inplace=True,
)
df_false_rejects.to_csv(os.path.join(SAVE_DIR, "false_rejects.csv"), index=False)

n = min(6, len(df_false_rejects))
rows = (n + 2) // 3
plt.figure(figsize=(10, 8))
for i in range(n):
    row = df_false_rejects.iloc[i]
    img1 = Image.open(row["Image 1: Path"])
    img2 = Image.open(row["Image 2: Path"])
    target_width = min(img1.width, img2.width)
    target_height = min(img1.height, img2.height)
    new_size = (target_width, target_height)

    img1_resized = img1.resize(new_size)
    img2_resized = img2.resize(new_size)

    combined = Image.new("RGB", (new_size[0] * 2, new_size[1]))
    combined.paste(img1_resized, (0, 0))
    combined.paste(img2_resized, (new_size[0], 0))
    ax = plt.subplot(rows, 3, i + 1)
    ax.imshow(combined)
    ax.axis("off")

    title_str = f"Genuine -> Rejected\nScore: {row['Similarity Score']:.2f}"
    ax.set_title(title_str, fontsize=10)
plt.suptitle("Unverified Genuine Pairs (False Rejects)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "false_rejects.png"), dpi=600)
plt.close()

df_false_accepts = pd.DataFrame(report["false_accepts"])
df_false_accepts.rename(
    columns={
        "path1": "Image 1: Path",
        "path2": "Image 2: Path",
        "score": "Similarity Score",
    },
    inplace=True,
)
df_false_accepts.to_csv(os.path.join(SAVE_DIR, "false_accepts.csv"), index=False)

n = min(6, len(df_false_accepts))
rows = (n + 2) // 3
plt.figure(figsize=(10, 8))
for i in range(n):
    row = df_false_accepts.iloc[i]
    img1 = Image.open(row["Image 1: Path"])
    img2 = Image.open(row["Image 2: Path"])
    target_width = min(img1.width, img2.width)
    target_height = min(img1.height, img2.height)
    new_size = (target_width, target_height)

    img1_resized = img1.resize(new_size)
    img2_resized = img2.resize(new_size)

    combined = Image.new("RGB", (new_size[0] * 2, new_size[1]))
    combined.paste(img1_resized, (0, 0))
    combined.paste(img2_resized, (new_size[0], 0))
    ax = plt.subplot(rows, 3, i + 1)
    ax.imshow(combined)
    ax.axis("off")

    title_str = f"Impostor -> Accepted\nScore: {row['Similarity Score']:.2f}"
    ax.set_title(title_str, fontsize=10)
plt.suptitle("Verified Impostor Pairs (False Accepts)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "false_accepts.png"), dpi=600)
plt.close()

rename_dict = {
    "feat_dim": "feature_dimensions",
    "dropout": "dropout_rate",
    "s": "arcface_scaling_factor",
    "m": "arcface_angular_margin_penalty",
    "lambda_c": "lambda_center",
    "lr": "learning_rate",
    "wd": "weight_decay",
    "lr_center": "learning_rate_center",
    "wd_center": "weight_decay_center",
    "momentum_center": "momentum_center",
    "T_0": "scheduler_T0",
    "eta_min": "scheduler_eta_min",
    "step_size_center": "scheduler_center_step_size",
    "gamma_center": "scheduler_center_gamma",
}
metrics = {}
best_params = {rename_dict.get(k, k): v for k, v in best_params.items()}
metrics["best_params"] = best_params
metrics["decision_thresholds"] = {
    "osr_decision_threshold": osr_threshold,
    "verification_decision_threshold": verif_threshold,
}
metrics["identification"] = {
    key: value
    for key, value in metrics_osr_id.items()
    if key.startswith("identification_") and key != "identification_per_class_metrics"
}
metrics["open_set"] = {
    key: value for key, value in metrics_osr_id.items() if key.startswith("open_set_")
}
metrics["verification"] = metrics_verif
with open(os.path.join(SAVE_DIR, "final_results.json"), "w") as f:
    json.dump(metrics, f, indent=4)

os.remove(os.path.join(SAVE_DIR, "dataset.pkl"))
for i in range(1, 6):
    os.remove(os.path.join(SAVE_DIR, f"study_{i}.pkl"))
    os.remove(os.path.join(SAVE_DIR, f"outer_{i}.pkl"))

with zipfile.ZipFile(
    os.path.join(os.getcwd(), "weights.zip"), "w", zipfile.ZIP_DEFLATED
) as zip_ref:
    for file in os.listdir(SAVE_DIR):
        if file.endswith((".csv", ".json", ".pkl", ".png", ".pt")):
            file_path = os.path.join(SAVE_DIR, file)
            zip_ref.write(file_path, os.path.relpath(file_path, SAVE_DIR))
