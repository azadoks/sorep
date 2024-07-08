# %%
import imblearn.ensemble as imbe
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.ensemble as ske
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp

# %%
with open("../data/mc3d_dos_fermi_centered_gauss_0.05.npz", "rb") as fp:
    data = dict(np.load(fp))

N = data["features"].shape[0]
# %%
random_state = np.random.randint(0, 2**32 - 1)

X_train, X_test, y_train, y_test = skms.train_test_split(
    data["features"], data["meets_tcm_criteria"], train_size=0.10, random_state=random_state
)
# %%
scaler = skp.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# %%
rfc = imbe.BalancedRandomForestClassifier(
    n_estimators=400,
    random_state=random_state,
    class_weight="balanced",
    sampling_strategy="all",
    replacement=True,
    bootstrap=False,
)
# %%
rfc.fit(X_train, y_train)
# %%
y_train_pred = rfc.predict(X_train)
y_test_pred = rfc.predict(X_test)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

cm_train = skm.confusion_matrix(y_true=y_train, y_pred=y_train_pred)
skm.ConfusionMatrixDisplay(cm_train).plot(ax=axes[0])
axes[0].set_title("Train")

cm_test = skm.confusion_matrix(y_true=y_test, y_pred=y_test_pred)
skm.ConfusionMatrixDisplay(cm_test).plot(ax=axes[1])
axes[1].set_title("Test")
# %%
