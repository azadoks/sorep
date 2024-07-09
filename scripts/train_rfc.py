# %%
import imblearn.ensemble as imbe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as skpl
import sklearn.preprocessing as skpp

# %%  Load data
with open("../data/mc3d_targets.npz", "rb") as fp:
    npz = np.load(fp)
    target_df = pd.DataFrame(data=dict(npz))

with open("../data/mc3d_features_single_shot_dos_fermi_centered_gauss_0.05.npz", "rb") as fp:
    npz = np.load(fp)
    feature_df = pd.DataFrame(data={k: v.tolist() for (k, v) in npz.items()})

df = pd.merge(target_df, feature_df, on="material_id")
X = np.array(df["features"].tolist())
y = df["meets_tcm_criteria"].to_numpy()

# Create constant hold-out validation set of 10% of the data
X, X_val, y, y_val = skms.train_test_split(X, y, test_size=0.1, random_state=9997)

# %% Pre-process data
train_size = 0.01
random_state = np.random.randint(0, 2**32 - 1)
X_train, X_test, y_train, y_test = skms.train_test_split(X, y, train_size=train_size, random_state=random_state)
# %% Create pipeline
pipe = skpl.Pipeline(
    [
        ("scaler", skpp.StandardScaler()),
        (
            "gscv",
            skms.GridSearchCV(
                imbe.BalancedRandomForestClassifier(),
                param_grid={
                    "n_estimators": [500],
                    "class_weight": ["balanced", "balanced_subsample"],
                    "sampling_strategy": ["all", "minority", "not minority", "not majority", "all"],
                    "replacement": [True, False],
                    "bootstrap": [True, False],
                    "ccp_alpha": [0e-0, 1e-3, 5e-3, 1e-2],
                    "random_state": [0],
                },
                n_jobs=-1,
            ),
        ),
    ]
)
# %% Fit and evaluate
pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

cm_train = skm.confusion_matrix(y_true=y_train, y_pred=y_train_pred)
skm.ConfusionMatrixDisplay(cm_train).plot(ax=axes[0])
axes[0].set_title(f"Train (f={train_size}|N={len(y_train)})")

cm_test = skm.confusion_matrix(y_true=y_test, y_pred=y_test_pred)
skm.ConfusionMatrixDisplay(cm_test).plot(ax=axes[1])
axes[1].set_title(f"Test (f={1-train_size}|N={len(y_test)})")
# %%
