import os
import pickle

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

from utils import (
    SAVE_DIR,
    get_dataloader,
    train_model,
    build_prototype_gallery,
    extract_embeddings,
    tune_osr_threshold,
    tune_pair_threshold,
    evaluate_osr_identification,
    evaluate_verification,
)

with open(os.path.join(SAVE_DIR, "dataset.pkl"), "rb") as f:
    data = pickle.load(f)

num_known = data["num_known"]
unknown_val = data["unknown_val"]

with open(os.path.join(SAVE_DIR, "outer_5.pkl"), "rb") as f:
    data = pickle.load(f)

outer_train = data["outer_train"]
outer_train_labels = data["outer_train_labels"]

device = "cuda"
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)


def objective(trial):
    trial.suggest_categorical("feat_dim", [256, 384, 512])
    trial.suggest_float("dropout", 0.2, 0.5)
    trial.suggest_int("s", 45, 65)
    trial.suggest_float("m", 0.35, 0.45)
    trial.suggest_float("lambda_c", 1e-4, 1e-1)

    trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    trial.suggest_float("wd", 1e-6, 1e-3, log=True)

    trial.suggest_float("lr_center", 1e-5, 5e-4, log=True)
    trial.suggest_float("wd_center", 1e-6, 1e-3, log=True)
    trial.suggest_float("momentum_center", 0.8, 0.95)

    trial.suggest_int("T_0", 3, 10)
    trial.suggest_float("eta_min", 1e-6, 1e-4)

    trial.suggest_int("step_size_center", 5, 20)
    trial.suggest_float("gamma_center", 0.1, 0.9)

    scores = []
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
    for inner_idx, (train_idx, val_idx) in enumerate(
        inner_cv.split(outer_train, outer_train_labels)
    ):
        inner_train = [outer_train[i] for i in train_idx]
        inner_val = [outer_train[i] for i in val_idx]

        train_loader = get_dataloader(inner_train, "train")
        known_val_loader = get_dataloader(inner_val, "val")
        pair_val_loader = get_dataloader(
            inner_val + unknown_val, "val", num_pairs=10000
        )
        unknown_val_loader = get_dataloader(unknown_val, "val")

        model = train_model(
            train_loader, known_val_loader, num_known, device, trial.params
        )
        gallery = build_prototype_gallery(
            extract_embeddings(model, device, train_loader), num_prototypes=3
        )

        osr_threshold = tune_osr_threshold(
            model,
            gallery,
            known_val_loader,
            unknown_val_loader,
            device,
            plot=False,
        )
        verif_threshold = tune_pair_threshold(
            model, pair_val_loader, device, plot=False
        )

        metrics_osr_id, _ = evaluate_osr_identification(
            model,
            gallery,
            known_val_loader,
            unknown_val_loader,
            device,
            osr_threshold,
            plot=False,
        )
        metrics_verif, _ = evaluate_verification(
            model, pair_val_loader, device, verif_threshold, plot=False
        )

        score = (
            0.34 * metrics_osr_id["identification_accuracy_closed_set"]
            + 0.33 * metrics_osr_id["open_set_auroc"]
            + 0.33 * metrics_verif["verification_auroc"]
        )
        scores.append(score)

        trial.report(score, inner_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


sampler = TPESampler()
pruner = MedianPruner(n_startup_trials=10)
study = optuna.create_study(
    study_name="Model Hyperparameter Tuning: Fold 5",
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

with open(os.path.join(SAVE_DIR, "study_5.pkl"), "wb") as f:
    pickle.dump(study, f)
