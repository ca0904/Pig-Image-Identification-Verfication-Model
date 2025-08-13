import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    IMAGE_DIR,
    SAVE_DIR,
    filter_dataset,
    prepare_dataset_splits,
    prepare_unknown_splits,
    cross_validate,
)

filtered_dataset = filter_dataset(IMAGE_DIR)
known_classes, unknown_classes = prepare_dataset_splits(IMAGE_DIR)
unknown_val, unknown_test = prepare_unknown_splits(filtered_dataset, unknown_classes)
num_known = len(known_classes)

dist_df = pd.DataFrame(
    {
        "Label": list(filtered_dataset.keys()),
        "Support": [len(images) for images in filtered_dataset.values()],
    }
).set_index("Label")

plt.figure(figsize=(10, 5))
ax = plt.gca()
dist_df.plot(
    kind="bar", legend=False, ax=ax
)
plt.ylabel("Number of Images")
plt.xlabel("Class Label")
ax.set_xticklabels(dist_df.index, fontsize=10)
plt.title("Dataset Distribution per Class")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dataset_distribution.png"), dpi=600)
plt.close()

with open(os.path.join(SAVE_DIR, "dataset.pkl"), "wb") as f:
    pickle.dump(
        {
            "filtered_dataset": filtered_dataset,
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "num_known": num_known,
            "unknown_val": unknown_val,
            "unknown_test": unknown_test,
        },
        f,
    )

cross_validate(filtered_dataset, known_classes)
