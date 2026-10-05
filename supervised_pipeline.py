# ---------------------- CONFIG ----------------------
H5AD_PATH       = "single_cell.h5ad"
TRAIN_CSV       = "singlecell_train.csv"
FIG_DIR         = "figs"
OUT_DIR         = "supervised_outputs"
SUBMISSION_DIR  = "submissions"

TRAIN_ID_COL    = "ID"
TRAIN_LABEL_COL = "cell_type"

# Core sweeps
HVG_COUNTS      = [250, 1000, 1250, 1500, 1750, 2000, 4000]

# Biology filters (builds separate variants)
BIO_FILTER_STAGE   = ["pre", "post"]
BIO_FILTER_LEVELS  = ["none", "mild", "medium", "aggressive"]

AE_LATENT_GRID   = [8, 12, 16, 32, 64]
AE_EPOCHS        = 60
AE_BATCH_SIZE    = 64

# Supervised search (rigorous only)
ALWAYS_RIGOROUS  = True  # supervised evaluation reselects HVG & PCA per-fold (no leakage)
OUTER_K          = 8
INNER_K          = 4

# Practical guardrails (optional)
MAX_VARIANTS_HINT = None
FINAL_SUBMISSION_CSV = "submission_final.csv"

# ---------------------- RUNTIME PROFILE FLAGS ----------------------
DEBUG_MODE     = False
USE_GPU        = True
GPU_PCA        = True

# --- Supervised search budget & strategy ---
SUPERVISED_MAX_FITS = 15000         # hard cap on model.fit() calls in section 8
SUPERVISED_MIN_TRIALS = 32         # at least this many trials, even if budget is tiny
RANDOM_STATE = 42


# Debug overrides
if DEBUG_MODE:
    HVG_COUNTS = [1000, 2000]
    BIO_FILTER_LEVELS = ["none", "mild"]

    AE_LATENT_GRID = [8, 16]
    AE_EPOCHS      = 10
    AE_BATCH_SIZE  = 128

    OUTER_K         = 5
    INNER_K         = 3

    MAX_VARIANTS_HINT = 10



# ---------------------- Imports ----------------------
import os, json, math, warnings, sys, gc, itertools, random, time, re, shutil
import types as _types
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
import json
try:
    import scanpy as sc
    import anndata as ad
except Exception as e:
    raise RuntimeError("Requires scanpy/anndata.") from e
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
    raise RuntimeError("PyTorch is required for AE / GPU MLP / GPU PCA")
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception as e:
    raise RuntimeError("Requires xgboost")
    HAVE_XGB = False
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, accuracy_score, f1_score, log_loss,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
import inspect
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestRegressor
try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    HAVE_CUPY = False
try:
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.linear_model import LogisticRegression as cuLogReg
    from cuml.svm import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAVE_CUML = True
except Exception:
    HAVE_CUML = False
from contextlib import contextmanager
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kw):
        return iterable if iterable is not None else range(0)

# Torch device
if HAVE_TORCH:
    TORCH_DEVICE = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
else:
    TORCH_DEVICE = "cpu"
from joblib import Memory
PIPELINE_CACHE = None

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=120, frameon=False)

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ---------------------- helpers ----------------------
T0 = time.perf_counter()

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    """Print with absolute+elapsed timestamps."""
    print(f"[{_ts()}  +{time.perf_counter()-T0:6.1f}s] {msg}")

@contextmanager
def section(title):
    """Mark major steps and show step elapsed time."""
    log(f"▶ {title} ...")
    t = time.perf_counter()
    try:
        yield
    finally:
        log(f"✔ {title} done in {time.perf_counter()-t:.1f}s")

def _clean_jsonable(v):
    import numpy as _np
    if isinstance(v, (set,)):
        return sorted(list(v))
    if isinstance(v, (_np.integer, _np.floating)):
        return v.item()
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            return str(v)
    if isinstance(v, dict):
        return {str(k): _clean_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_jsonable(x) for x in v]
    return str(v)

def dump_run_params(globs: dict):
    """Dump every ALL-CAPS *config-like* variable defined in this module."""
    keys = sorted(k for k in globs.keys() if k.isupper() and not k.startswith("_"))
    params = {}
    for k in keys:
        v = globs[k]
        if isinstance(v, (_types.ModuleType, type)) or callable(v):
            continue
        params[k] = _clean_jsonable(v)

    print("\n===== RUN PARAMETERS =====")
    print(json.dumps(params, indent=2))
    print("==========================\n")
    return params

def _fmt_dur(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# ---------------------------------------------------------------------------
def _to_dense(X):
    if hasattr(X, "A"):
        return X.A
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

def symbol_series_upper(adata):
    idx = adata.var_names.astype(str)
    if "Symbol" in adata.var.columns:
        s = pd.Series(adata.var["Symbol"], index=idx)
    else:
        s = pd.Series(idx, index=idx)
    s = s.astype("string").fillna("")
    return s.str.upper()

def gene_filter_mask(adata, level="mild"):
    su = symbol_series_upper(adata)
    up = su.values

    mito = np.array([s.startswith(("MT-", "MT_", "MT.")) for s in up])
    ribo = np.array([s.startswith(("RPL", "RPS", "MRPL", "MRPS")) for s in up])
    hemo = np.array([s.startswith(("HBA", "HBB", "HBD", "HBM", "HBZ", "HBE", "HBG")) for s in up])
    ig   = np.array([s.startswith(("IGH", "IGL", "IGK", "TRB", "TRA", "TRG", "TRD")) for s in up])

    S_genes   = set([g.upper() for g in ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]])
    G2M_genes = set([g.upper() for g in ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"]])
    cc  = su.isin(S_genes | G2M_genes).values
    sex = np.array([s in {"XIST","XACT","RPS4Y1","DDX3Y","KDM5D","EIF1AY","UTY"} for s in up])

    keep = np.ones(len(up), dtype=bool)
    if level == "none":
        return keep
    if level == "mild":
        keep = ~(mito | ribo)
    elif level == "medium":
        keep = ~(mito | ribo | hemo | ig)
    elif level == "aggressive":
        keep = ~(mito | ribo | hemo | ig | cc | sex)
        Xd = _to_dense(adata.X)
        means = np.asarray(Xd.mean(axis=0)).ravel()
        vars_ = np.asarray(Xd.var(axis=0)).ravel()
        hi_mean = means >= np.quantile(means, 0.95)
        lo_var  = vars_ <= np.quantile(vars_, 0.20)
        keep = keep & ~(hi_mean & lo_var)
    return keep

def add_counts_to_boxplot(ax, data, labels, y_pad_ratio=0.02, fontsize=8):
    non_empty = [np.asarray(d) for d in data if len(d) > 0]
    if not non_empty:
        return

    all_vals = np.concatenate(non_empty)
    y_min = float(all_vals.min())
    y_max = float(all_vals.max())
    y_range = y_max - y_min if y_max > y_min else 1.0
    y_offset = y_pad_ratio * y_range

    ax.set_ylim(top=y_max + 2 * y_offset)

    for i, d in enumerate(data, start=1):
        d = np.asarray(d)
        n = len(d)
        if n == 0:
            continue
        y = float(d.max())
        ax.text(
            i,
            y + y_offset,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

def plot_confusion_from_preds(y_true_enc, y_pred_enc, class_names, out_path_prefix):
    n_classes = len(class_names)
    cm = confusion_matrix(
        y_true_enc,
        y_pred_enc,
        labels=np.arange(n_classes),
    )
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fraction of true label", rotation=90)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            frac = cm_norm[i, j]
            text = f"{count}\n{frac*100:.1f}%"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(f"{out_path_prefix}.png", dpi=150)
    plt.close(fig)

def cv_predictions_for_trial(trial_spec,
                             model_spaces,
                             X_lab_df,
                             y_lab_enc,
                             outer_k=OUTER_K,
                             random_state=RANDOM_STATE):
    """
    Executes a StratifiedKFold CV with fixed hyperparameters (trial_spec)
    and yields all the OOF predictions.
    Builds trustworthy metrics for individual models
    """
    cv = StratifiedKFold(
        n_splits=outer_k,
        shuffle=True,
        random_state=random_state,
    )
    all_true = []
    all_pred = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_lab_df, y_lab_enc), 1):
        X_tr = X_lab_df.iloc[tr_idx]
        X_te = X_lab_df.iloc[te_idx]
        y_tr = y_lab_enc[tr_idx]
        y_te = y_lab_enc[te_idx]

        hvgt_step = ("hvgpca", HVGPCATransformer(
            n_top_hvg=trial_spec["n_top_hvg"],
            stage=trial_spec["stage"],
            filter_level=trial_spec["filter_level"],
            regress_cc=trial_spec["regress_cc"],
            n_pcs=trial_spec["n_pcs"],
            use_gpu_pca=GPU_PCA,
            repr_kind=trial_spec["repr_kind"],
            ae_latent_dim=trial_spec["ae_latent_dim"],
            ae_epochs=AE_EPOCHS,
            ae_batch_size=AE_BATCH_SIZE,
            use_marker_scores=trial_spec.get("use_marker_scores", False),
        ))

        spec_model = model_spaces[trial_spec["model"]]
        base_est = spec_model["make_estimator"](trial_spec["model_params"])
        if isinstance(base_est, Pipeline):
            est = Pipeline([hvgt_step] + base_est.steps, memory=PIPELINE_CACHE)
        else:
            est = Pipeline([hvgt_step, ("clf", base_est)], memory=PIPELINE_CACHE)

        sw_tr = compute_balanced_sample_weights(y_tr)
        fit_params = build_fit_params_with_sample_weight(est, sw_tr)
        est.fit(X_tr, y_tr, **fit_params)

        y_pred_fold = est.predict(X_te)
        all_true.append(y_te)
        all_pred.append(y_pred_fold)

    return np.concatenate(all_true), np.concatenate(all_pred)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple PyTorch MLP with a scikit-learn compatible API.
    Uses CrossEntropyLoss and Adam, runs on TORCH_DEVICE.
    Uses grids:
      - hidden_layer_sizes
      - alpha (L2 weight decay)
      - learning_rate_init
      - batch_size
      - max_iter
      - early_stopping / n_iter_no_change
    """
    def __init__(self,
                 hidden_layer_sizes=(128,),
                 activation="relu",
                 dropout=0.0,
                 alpha=1e-4,
                 learning_rate_init=1e-3,
                 batch_size=64,
                 max_iter=200,
                 tol=1e-4,
                 early_stopping=True,
                 n_iter_no_change=10,
                 random_state=0,
                 device=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout = float(dropout)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.device = device or TORCH_DEVICE

    def _build_model(self, n_features, n_classes):
        layers = []
        in_dim = n_features
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            if self.activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        return nn.Sequential(*layers)


    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X_np.shape

        # Encode classes (0..n_classes-1) and store mapping
        classes, y_enc = np.unique(y_np, return_inverse=True)
        self.classes_ = classes
        n_classes = len(self.classes_)

        device = torch.device(self.device)
        model = self._build_model(n_features, n_classes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )

        # -------- Class-balanced weights for CrossEntropyLoss --------
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=y_enc,
        )
        class_weights = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # ------------------------------------------------------------------

        ds = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_enc),
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.loss_curve_ = []
        best_loss = np.inf
        best_state = None
        no_improve = 0

        for epoch in range(self.max_iter):
            model.train()
            total_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / n_samples
            self.loss_curve_.append(avg_loss)

            if self.early_stopping:
                if avg_loss + self.tol < best_loss:
                    best_loss = avg_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        break

        if best_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_state.items()}
            )
        self.model_ = model
        self.n_features_in_ = n_features
        return self


    def _predict_proba_internal(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPClassifier is not fitted yet.")
        X_np = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device)
        X_t = torch.from_numpy(X_np).to(device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_proba(self, X):
        return self._predict_proba_internal(X)

    def predict(self, X):
        probs = self._predict_proba_internal(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]

# ---------------------- BOOTSTRAP + PARAM DUMP ----------------------
log("Bootstrapping & parameter dump")
_ = dump_run_params(globals())

# ---------------------- 1) Load & align IDs ----------------------
with section("1) Load & align IDs"):
    train_df = pd.read_csv(TRAIN_CSV)
    assert TRAIN_ID_COL in train_df.columns and TRAIN_LABEL_COL in train_df.columns, \
        f"Expected columns ({TRAIN_ID_COL}, {TRAIN_LABEL_COL}) in {TRAIN_CSV}"

    adata = sc.read_h5ad(H5AD_PATH)
    adata.var_names = adata.var_names.astype(str)
    if "Symbol" in adata.var.columns:
        adata.var["Symbol"] = adata.var["Symbol"].astype("string").fillna("")
    adata.obs["cell_id"] = (
        adata.obs["ID"].astype(str)
        if "ID" in adata.obs.columns
        else adata.obs_names.astype(str)
    )
    id2lab = dict(zip(train_df[TRAIN_ID_COL].astype(str),
                      train_df[TRAIN_LABEL_COL].astype(str)))
    adata.obs["train_label"] = adata.obs["cell_id"].map(id2lab)
    adata.obs["is_labeled"] = adata.obs["train_label"].notna()
    labeled_ids = set(train_df[TRAIN_ID_COL].astype(str))
    adata.obs["is_unlabeled_target"] = ~adata.obs["cell_id"].isin(labeled_ids)

    print(adata)
    print("Labeled cells in h5ad:", int(adata.obs["is_labeled"].sum()))
    print("Unlabeled target cells:", int(adata.obs["is_unlabeled_target"].sum()))

# ---------------------- 2) QC ----------------------
with section("2) QC"):
    Xdense = _to_dense(adata.X)
    adata.obs["nnz_genes"] = (Xdense > 0).sum(axis=1)
    adata.obs["mean_expr"] = Xdense.mean(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(adata.obs["nnz_genes"], bins=40)
    ax[0].set_title("Non-zero genes per cell")
    ax[1].hist(adata.obs["mean_expr"], bins=40)
    ax[1].set_title("Mean expression per cell")
    for a in ax:
        a.set_xlabel("value"); a.set_ylabel("cells")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/qc_histograms.png", dpi=150)
    plt.close()
    log("Saved QC histograms → figs/qc_histograms.png")
# ---------------------- 3) Supervised RANDOM search (HVG + optional PCA, guided budget) ----------------------
#
# We replace the huge nested grid-search with a *budgeted random search*:
# - Each "trial" picks a biology filter spec, HVG count, #PCs (or 0=HVG-only),
#   a model family, and a concrete hyperparameter set.
# - That trial is evaluated with OUTER_K-fold CV (no inner CV).
# - Total number of fits is capped by SUPERVISED_MAX_FITS.
# - Trials are sampled in a balanced way so that each value of
#   (bio_tag, HVG, n_pcs, model, and each model hyperparameter option)
#   is used roughly the same number of times, which keeps the plots meaningful.
SUPERVISED_RANDOM_SEED = RANDOM_STATE
log(f"Supervised search budget: max {SUPERVISED_MAX_FITS} fits")
def get_population_marker_sets():
    """
    Hand-crafted marker gene sets for each of the 9 main 'broad' classes
    (excluding the 'other' catch-all class).

    Gene symbols are returned as upper-case strings and are independent
    of the current dataset (no label leakage).

    NOTE:
      - MPP1/2/3 markers are heuristic combinations of other programs
        (myeloid / erythroid / lymphoid-biased). You can refine these
        from biology as you like.
    """
    base = {
        # Long-term / primitive HSC program
        "HSC_like": [
            "KIT", "LY6A", "PROCR", "HLF", "MECOM",
            "GATA2", "MEIS1", "TEK", "ANGPT1"
        ],
        # Lymphoid-primed multipotent progenitors
        "LMPP_like": [
            "FLT3", "DNTT", "IL7R", "SATB1"
        ],
        # Granulocyte–monocyte progenitors
        "GMP_like": [
            "ELANE", "MPO", "PRTN3", "CTSG",
            "S100A8", "S100A9"
        ],
        # Megakaryocyte–erythroid progenitors
        "MEP_like": [
            "KLF1", "GATA1", "HBB", "HBA1", "HBA2",
            "CAR1", "CAR2", "PF4", "PPBP",
            "ITGA2B", "VWF"
        ],
        # Classical CMP program
        "CMP_like": [
            "SPI1", "CEBPB",
            "CSF1R", "CSF3R",
            "IRF8"
        ],
        # Short-term / activated HSC program
        "STHSC_like": [
            "CD34", "MPL", "THPO", "CXCR4", "PROM1"
        ],
    }

    for k, genes in base.items():
        base[k] = sorted({str(g).upper() for g in genes})

    # Map into the 9 supervised 'broad' classes
    pop_markers = {
        # Direct mappings
        "MEP_broad":   base["MEP_like"],
        "CMP_broad":   base["CMP_like"],
        "LMPP_broad":  base["LMPP_like"],
        "GMP_broad":   base["GMP_like"],
        "LTHSC_broad": sorted(set(base["HSC_like"] + base["STHSC_like"])),
        "STHSC_broad": base["STHSC_like"],

        # Heuristic combinations for MPPs:
        #   - MPP1: myeloid-biased (CMP + GMP)
        #   - MPP2: erythroid/megakaryocyte-biased (CMP + MEP)
        #   - MPP3: more lymphoid/early (CMP + LMPP)
        "MPP1_broad":  sorted(set(base["CMP_like"] + base["GMP_like"])),
        "MPP2_broad":  sorted(set(base["CMP_like"] + base["MEP_like"])),
        "MPP3_broad":  sorted(set(base["CMP_like"] + base["LMPP_like"])),
    }

    for k, genes in pop_markers.items():
        pop_markers[k] = sorted({str(g).upper() for g in genes})

    return pop_markers


# ---------------------- HVG + representation transformer for supervised CV ----------------------
class HVGPCATransformer(BaseEstimator, TransformerMixin):
    """
    Leak-free HVG + representation transformer for supervised CV.

    All fitting is done only on the training fold:

      1. Select HVGs on training cells only.
      2. Optional biology filters (pre / post HVG).
      3. Optional cell-cycle regression (S / G2M).
      4. Standardize genes (per-gene mean/std from training).
      5. Build representation:

         repr_kind == "hvg"        -> scaled HVGs (no PCA)
         repr_kind == "pca"        -> PCA
         repr_kind == "pca_whiten" -> PCA with whitening
         repr_kind == "ae"         -> AE latent codes (PyTorch).
    """

    def __init__(
        self,
        n_top_hvg=2000,
        stage="pre",
        filter_level="none",
        regress_cc=False,
        n_pcs=50,
        use_gpu_pca=True,
        repr_kind="pca",          # "hvg" | "pca" | "pca_whiten" | "ae"
        ae_latent_dim=16,
        ae_epochs=AE_EPOCHS,
        ae_batch_size=AE_BATCH_SIZE,
        use_marker_scores=False,
    ):
        self.n_top_hvg = int(n_top_hvg)
        self.stage = stage
        self.filter_level = filter_level
        self.regress_cc = bool(regress_cc)
        self.n_pcs = int(n_pcs)
        self.use_gpu_pca = bool(use_gpu_pca)

        self.repr_kind = str(repr_kind)
        self.ae_latent_dim = int(ae_latent_dim)
        self.ae_epochs = int(ae_epochs)
        self.ae_batch_size = int(ae_batch_size)
        self.use_marker_scores = bool(use_marker_scores)

    # ---------------- internal helpers ----------------
    def _cell_indexer(self, cell_ids):
        """Map cell_ids (strings) to row indices in global `adata`."""
        cell_ids = np.asarray(cell_ids, dtype=str)
        all_ids = pd.Index(adata.obs["cell_id"].astype(str).values)
        idx = all_ids.get_indexer(cell_ids)
        if (idx < 0).any():
            missing = cell_ids[idx < 0]
            raise ValueError(
                f"HVGPCATransformer: some cell IDs not found in adata: "
                f"{missing[:5]}..."
            )
        return idx

    def _build_cc_masks(self, var_idx):
        """
        Build boolean masks over the HVG gene axis for S-phase and G2M-phase genes.
        var_idx are indices into adata.var (global); masks are length = n_hvg_genes.
        """
        su_full = symbol_series_upper(adata)
        su_hvg = su_full.iloc[var_idx].values

        S_genes = set([
            "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
            "CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1",
            "GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51",
            "RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1",
            "CLSPN","POLA1","CHAF1B","BRIP1","E2F8"
        ])
        G2M_genes = set([
            "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
            "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
            "CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1",
            "KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1",
            "NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1",
            "ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"
        ])

        su_hvg = np.asarray(su_hvg, dtype=str)
        s_mask = np.array([g in S_genes for g in su_hvg], dtype=bool)
        g2m_mask = np.array([g in G2M_genes for g in su_hvg], dtype=bool)
        return s_mask, g2m_mask

    def _regress_cc(self, X, s_mask, g2m_mask):
        """
        Regress out cell-cycle covariates (S and G2M mean expression) from each gene.
        X : (n_cells, n_genes) matrix.
        Returns residuals and stores regression coefficients.
        """
        n_cells, n_genes = X.shape
        if n_cells < 3:
            self.cc_betas_ = None
            return X

        C = np.zeros((n_cells, 2), dtype=np.float32)
        if s_mask.any():
            C[:, 0] = X[:, s_mask].mean(axis=1)
        if g2m_mask.any():
            C[:, 1] = X[:, g2m_mask].mean(axis=1)

        Z = np.concatenate(
            [np.ones((n_cells, 1), dtype=np.float32), C],
            axis=1
        )
        # Solve least squares for all genes at once:
        ZTZ = Z.T @ Z          # (3, 3)
        ZTX = Z.T @ X          # (3, n_genes)
        try:
            betas = np.linalg.solve(ZTZ, ZTX).astype(np.float32)
        except np.linalg.LinAlgError:
            self.cc_betas_ = None
            return X

        self.cc_betas_ = betas   # store for transform()
        X_resid = X - Z @ betas  # (n_cells, n_genes)
        return X_resid

    # -------- marker-score helpers  --------
    def _init_marker_indices(self):
        """
        Map each population marker set to global adata.var indices.
        This is done once at fit-time and reused at transform-time.
        """
        pop_markers = get_population_marker_sets()
        self.marker_pop_names_ = list(pop_markers.keys())

        su = symbol_series_upper(adata)
        su_vals = np.asarray(su.values, dtype=str)

        sym_to_idx = {}
        for j, sym in enumerate(su_vals):
            sym = str(sym).upper()
            sym_to_idx.setdefault(sym, []).append(j)

        marker_var_idx = {}
        for pop, genes in pop_markers.items():
            idxs = []
            for g in genes:
                g = str(g).upper()
                idxs.extend(sym_to_idx.get(g, []))
            idxs = sorted(set(idxs))
            marker_var_idx[pop] = np.asarray(idxs, dtype=int)

        self.marker_var_idx_ = marker_var_idx

    def _marker_scores_for_cells(self, obs_idx):
        """
        Compute per-cell marker scores for all configured populations.
        obs_idx: integer indices into global adata.obs (rows).
        Returns: (n_cells, n_marker_sets) array with mean expression per set.
        """
        if not hasattr(self, "marker_var_idx_") or self.marker_var_idx_ is None:
            self._init_marker_indices()

        n_cells = len(obs_idx)
        pop_names = getattr(self, "marker_pop_names_", [])
        n_sets = len(pop_names)

        scores = np.zeros((n_cells, n_sets), dtype=np.float32)

        for j, pop in enumerate(pop_names):
            idxs = self.marker_var_idx_.get(pop, None)
            if idxs is None or idxs.size == 0:
                continue
            X_sub = _to_dense(adata[obs_idx, idxs].X).astype(np.float32)
            scores[:, j] = X_sub.mean(axis=1)
        return scores

    # ---------------- fit/transform ----------------
    def fit(self, X, y=None):
        """
        Fit HVG selection, optional CC regression, scaling, and the chosen
        representation on *training* cells.

        - repr_kind="hvg"        → no PCA, returns scaled HVGs.
        - repr_kind="pca"        → PCA.
        - repr_kind="pca_whiten" → whitened PCA.
        - repr_kind="ae"         → AE latent codes.
        """
        if hasattr(X, "index"):
            cell_ids = X.index.astype(str).values
        else:
            cell_ids = np.asarray(X, dtype=str)

        # 1. Subset adata rows (cells)
        obs_idx = self._cell_indexer(cell_ids)
        A = adata[obs_idx].copy()   # cells × all genes

        # 2. Biology filter pre-HVG
        if self.stage == "pre" and self.filter_level != "none":
            keep = gene_filter_mask(A, level=self.filter_level)
            A = A[:, keep].copy()

        # 3. HVG selection
        sc.pp.highly_variable_genes(
            A,
            n_top_genes=self.n_top_hvg,
            flavor="seurat_v3",
            inplace=True,
        )
        A = A[:, A.var["highly_variable"].values].copy()

        # 4. Biology filter post-HVG
        if self.stage == "post" and self.filter_level != "none":
            keep = gene_filter_mask(A, level=self.filter_level)
            A = A[:, keep].copy()
        # 5. Map HVG genes back to global var index

        all_var_index = pd.Index(adata.var_names.astype(str).values)
        hvg_genes = A.var_names.astype(str).values
        var_idx = all_var_index.get_indexer(hvg_genes)
        if (var_idx < 0).any():
            missing = hvg_genes[var_idx < 0]
            raise ValueError(
                f"HVGPCATransformer: HVG genes not found in global adata: "
                f"{missing[:5]}..."
            )

        self.var_idx_ = var_idx.astype(int)
        self.var_names_ = hvg_genes
        self.n_genes_ = len(self.var_idx_)

        # 6. Build training matrix from global adata to avoid weird copies
        X_train = _to_dense(adata[obs_idx, self.var_idx_].X).astype(np.float32)

        # 7. Optional cell-cycle regression
        self.regress_cc_ = bool(self.regress_cc)
        if self.regress_cc_:
            s_mask, g2m_mask = self._build_cc_masks(self.var_idx_)
            self.cc_S_mask_ = s_mask
            self.cc_G_mask_ = g2m_mask
            X_train = self._regress_cc(X_train, s_mask, g2m_mask)
        else:
            self.cc_S_mask_ = None
            self.cc_G_mask_ = None
            self.cc_betas_ = None

        # 8. Standardize genes (train statistics)
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma[sigma < 1e-6] = 1.0

        self.mean_ = mu.astype(np.float32)
        self.std_ = sigma.astype(np.float32)
        X_scaled = (X_train - self.mean_) / self.std_

        # 9. Build representation
        rk = str(self.repr_kind).lower()
        self.repr_kind_ = rk

        self.pca_ = None
        self.ae_model_ = None
        self.n_pcs_eff_ = 0

        if rk == "hvg":
            self.n_features_out_ = X_scaled.shape[1]

        elif rk in ("pca", "pca_whiten"):
            # Decide number of components
            if self.n_pcs > 0:
                n_comps_eff = int(
                    min(self.n_pcs,
                        max(1, min(X_scaled.shape[0], X_scaled.shape[1]) - 1))
                )
            else:
                n_comps_eff = min(50, X_scaled.shape[1])  # default
            self.n_pcs_eff_ = n_comps_eff

            if rk == "pca_whiten":
                pca = PCA(
                    n_components=n_comps_eff,
                    whiten=True,
                    random_state=RANDOM_STATE,
                )
                Z = pca.fit_transform(X_scaled)
            else:
                if self.use_gpu_pca and HAVE_CUML and USE_GPU:
                    pca = cuPCA(
                        n_components=n_comps_eff,
                        random_state=RANDOM_STATE,
                    )
                    Z = pca.fit_transform(X_scaled)
                    try:
                        Z = cp.asnumpy(Z)
                    except Exception:
                        Z = np.asarray(Z)
                else:
                    pca = PCA(
                        n_components=n_comps_eff,
                        random_state=RANDOM_STATE,
                    )
                    Z = pca.fit_transform(X_scaled)

            self.pca_ = pca
            self.n_features_out_ = Z.shape[1]

        elif rk == "ae":
            if not HAVE_TORCH:
                raise RuntimeError("repr_kind='ae' requires PyTorch (HAVE_TORCH=False).")

            torch.manual_seed(RANDOM_STATE)
            device = torch.device(TORCH_DEVICE)

            class AE(nn.Module):
                def __init__(self, d_in, d_lat):
                    super().__init__()
                    self.enc = nn.Sequential(
                        nn.Linear(d_in, 512), nn.ReLU(),
                        nn.Linear(512, 256), nn.ReLU(),
                        nn.Linear(256, d_lat),
                    )
                    self.dec = nn.Sequential(
                        nn.Linear(d_lat, 256), nn.ReLU(),
                        nn.Linear(256, 512), nn.ReLU(),
                        nn.Linear(512, d_in),
                    )

                def forward(self, x):
                    z = self.enc(x)
                    xhat = self.dec(z)
                    return xhat, z

            X_np = np.asarray(X_scaled, dtype=np.float32)
            ds = TensorDataset(torch.from_numpy(X_np))
            dl = DataLoader(
                ds,
                batch_size=self.ae_batch_size,
                shuffle=True,
                pin_memory=(TORCH_DEVICE == "cuda"),
            )

            ae = AE(X_np.shape[1], self.ae_latent_dim).to(device)
            opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

            ae.train()
            for _ in range(self.ae_epochs):
                for (xb,) in dl:
                    xb = xb.to(device, non_blocking=True)
                    xhat, _z = ae(xb)
                    loss = ((xhat - xb) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            self.ae_model_ = ae.enc.to(device)
            self.ae_model_.eval()
            self.n_features_out_ = self.ae_latent_dim

        else:
            raise ValueError(f"Unknown repr_kind={rk!r}")

        # 10. Optional marker-score features
        self.use_marker_scores_ = bool(self.use_marker_scores)
        if self.use_marker_scores_:
            marker_scores = self._marker_scores_for_cells(obs_idx)
            m_mu = marker_scores.mean(axis=0).astype(np.float32)
            m_sigma = marker_scores.std(axis=0).astype(np.float32)
            m_sigma[m_sigma < 1e-6] = 1.0

            self.marker_mean_ = m_mu
            self.marker_std_ = m_sigma
            self.n_marker_features_ = marker_scores.shape[1]

            self.n_features_out_ = int(
                self.n_features_out_ + self.n_marker_features_
            )
        else:
            self.marker_mean_ = None
            self.marker_std_ = None
            self.marker_var_idx_ = None
            self.marker_pop_names_ = []
            self.n_marker_features_ = 0

        self.is_fitted_ = True
        return self


    def transform(self, X):
        """Apply the fitted pipeline to a new set of cells."""
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                "HVGPCATransformer must be fitted before calling transform()."
            )

        if hasattr(X, "index"):
            cell_ids = X.index.astype(str).values
        else:
            cell_ids = np.asarray(X, dtype=str)

        obs_idx = self._cell_indexer(cell_ids)

        # Extract the same genes (self.var_idx_) from global adata
        X_new = _to_dense(adata[obs_idx, self.var_idx_].X).astype(np.float32)

        # Apply same CC regression (if any)
        if self.regress_cc_ and self.cc_betas_ is not None:
            n_cells = X_new.shape[0]
            C = np.zeros((n_cells, 2), dtype=np.float32)
            if self.cc_S_mask_ is not None and self.cc_S_mask_.any():
                C[:, 0] = X_new[:, self.cc_S_mask_].mean(axis=1)
            if self.cc_G_mask_ is not None and self.cc_G_mask_.any():
                C[:, 1] = X_new[:, self.cc_G_mask_].mean(axis=1)

            Z_cov = np.concatenate(
                [np.ones((n_cells, 1), dtype=np.float32), C],
                axis=1
            )  # (n_cells, 3)
            X_new = X_new - Z_cov @ self.cc_betas_

        # Standardize using training stats
        X_scaled = (X_new - self.mean_) / self.std_

        rk = getattr(self, "repr_kind_", str(self.repr_kind).lower())

        # Base representation
        if rk == "hvg":
            Z = X_scaled

        elif rk in ("pca", "pca_whiten"):
            if self.pca_ is None:
                Z = X_scaled
            else:
                if (
                    rk == "pca"
                    and self.use_gpu_pca
                    and HAVE_CUML
                    and USE_GPU
                    and isinstance(self.pca_, cuPCA)
                ):
                    Z_new = self.pca_.transform(X_scaled)
                    try:
                        Z_new = cp.asnumpy(Z_new)
                    except Exception:
                        Z_new = np.asarray(Z_new)
                else:
                    Z_new = self.pca_.transform(X_scaled)
                Z = Z_new

        elif rk == "ae":
            if self.ae_model_ is None:
                raise RuntimeError(
                    "AE encoder not available in HVGPCATransformer."
                )
            device = torch.device(TORCH_DEVICE)
            X_np = np.asarray(X_scaled, dtype=np.float32)
            ds = TensorDataset(torch.from_numpy(X_np))
            dl = DataLoader(
                ds,
                batch_size=self.ae_batch_size,
                shuffle=False,
                pin_memory=(TORCH_DEVICE == "cuda"),
            )

            zs = []
            self.ae_model_.eval()
            with torch.no_grad():
                for (xb,) in dl:
                    xb = xb.to(device, non_blocking=True)
                    z = self.ae_model_(xb)
                    zs.append(z.cpu().numpy())
            Z = np.concatenate(zs, axis=0)

        else:
            Z = X_scaled
        if getattr(self, "use_marker_scores_", False):
            marker_scores = self._marker_scores_for_cells(obs_idx)
            if self.marker_mean_ is not None and self.marker_std_ is not None:
                marker_scores = (
                    marker_scores - self.marker_mean_
                ) / self.marker_std_
            Z = np.concatenate(
                [Z, marker_scores.astype(np.float32)], axis=1
            )

        return Z


# Helper to build biology-filter specs
def _build_bio_specs():
    specs = []
    for stage in BIO_FILTER_STAGE:
        for lvl in BIO_FILTER_LEVELS:
            specs.append(dict(
                tag=f"{stage}_{lvl}",
                stage=stage,
                level=lvl,
                regress_cc=False,
            ))
    # "clean_cc" special case (pre+mild+regress_cc=True)
    specs.append(dict(
        tag="clean_cc",
        stage="pre",
        level="mild",
        regress_cc=True,
    ))
    return specs

def _pc_grid_for(n_hvg: int) -> list[int]:
    if n_hvg <= 1000:
        return [10, 15, 20, 30]
    if n_hvg <= 1500:
        return [15, 20, 30, 40]
    return [20, 30, 40, 50]

def _pick_with_min_counts(counts: dict, choices: list):
    """
    Balanced sampler:
    among `choices`, pick those with the *lowest* usage count so far,
    then break ties uniformly at random.
    """
    pairs = [(v, counts[v]) for v in choices]
    minc = min(c for _, c in pairs)
    candidates = [v for v, c in pairs if c == minc]
    return random.choice(candidates)

def model_search_spaces(n_classes: int):
    spaces = {}

    # ----- MLP (Torch GPU if available, else sklearn MLP) -----
    if HAVE_TORCH and USE_GPU and TORCH_DEVICE == "cuda":
        def make_mlp(hp):
            clf = TorchMLPClassifier(
                hidden_layer_sizes=hp["hidden_layer_sizes"],
                activation=hp["activation"],
                dropout=hp["dropout"],
                alpha=hp["alpha"],
                learning_rate_init=hp["learning_rate_init"],
                batch_size=64,
                max_iter=600,
                n_iter_no_change=25,
                random_state=RANDOM_STATE,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ], memory=PIPELINE_CACHE)
    else:
        def make_mlp(hp):
            clf = MLPClassifier(
                hidden_layer_sizes=hp["hidden_layer_sizes"],
                alpha=hp["alpha"],
                learning_rate_init=hp["learning_rate_init"],
                activation=hp["activation"],
                batch_size=64,
                max_iter=600,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=RANDOM_STATE,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ], memory=PIPELINE_CACHE)

    spaces["mlp"] = dict(
        make_estimator=make_mlp,
        hyper_choices={
            "hidden_layer_sizes": [
                # 1-layer (shallow)
                (64,),
                (128,),
                (256,),
                (512,),
                # 2-layer
                (256, 128),
                (256, 256),
                (512, 256),
                (512, 512),
                # 3-layer
                (256, 256, 128),
                (512, 256, 128),
                (512, 512, 256),
                # 4-layer
                (256, 256, 256, 128),
                (512, 512, 256, 128)
            ],
            "activation": ["relu", "tanh"],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "alpha": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
            "learning_rate_init": [3e-4, 5e-4, 1e-3],
        },
    )

    # ----- Logistic Regression -----
    if HAVE_CUML and USE_GPU:
        def make_logreg(hp):
            clf = cuLogReg(
                C=hp["C"],
                max_iter=6000,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ], memory=PIPELINE_CACHE)
        logreg_choices = {
            "C": [0.25, 0.5, 1.0, 2.0, 4.0],
        }
    else:
        def make_logreg(hp):
            penalty = hp["penalty"]
            solver = "liblinear" if penalty == "l1" else "lbfgs"
            clf = LogisticRegression(
                C=hp["C"],
                penalty=penalty,
                class_weight=hp["class_weight"],
                solver=solver,
                max_iter=6000,
                random_state=RANDOM_STATE,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ], memory=PIPELINE_CACHE)
        logreg_choices = {
            "C": [0.25, 0.5, 1.0, 2.0, 4.0],
            "class_weight": [None, "balanced"],
            "penalty": ["l2", "l1"],
        }

    spaces["logreg"] = dict(
        make_estimator=make_logreg,
        hyper_choices=logreg_choices,
    )

    # ----- RBF SVM -----
    if HAVE_CUML and USE_GPU:
        def make_svm_rbf(hp):
            clf = cuSVC(
                kernel="rbf",
                C=hp["C"],
                gamma=hp["gamma"],
                probability=True,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ], memory=PIPELINE_CACHE)
        svm_choices = {
            "C": [0.5, 1.0, 3.0, 10.0],
            "gamma": [0.03, 0.01, 0.003],
        }
    else:
        def make_svm_rbf(hp):
            clf = SVC(
                kernel="rbf",
                C=hp["C"],
                gamma=hp["gamma"],
                class_weight=hp["class_weight"],
                probability=True,
                random_state=RANDOM_STATE,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ], memory=PIPELINE_CACHE)
        svm_choices = {
            "C": [0.5, 1.0, 3.0, 10.0],
            "gamma": ["scale", "auto", 0.03, 0.01],
            "class_weight": [None, "balanced"],
        }

    spaces["svm_rbf"] = dict(
        make_estimator=make_svm_rbf,
        hyper_choices=svm_choices,
    )

    # ----- Linear SVM  -----
    def make_linear_svm(hp):
        clf = LinearSVC(
            C=hp["C"],
            class_weight=hp["class_weight"],
            random_state=RANDOM_STATE,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ], memory=PIPELINE_CACHE)

    spaces["linear_svm"] = dict(
        make_estimator=make_linear_svm,
        hyper_choices={
            "C": [0.25, 0.5, 1.0, 2.0, 4.0],
            "class_weight": [None, "balanced"],
        },
    )

    # ----- Random Forest -----
    if HAVE_CUML and USE_GPU:
        def make_rf(hp):
            clf = cuRF(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                random_state=RANDOM_STATE,
            )
            return clf
        rf_choices = {
            "n_estimators": [200, 400, 800],
            "max_depth": [10, 20, 40],
        }
    else:
        def make_rf(hp):
            clf = RandomForestClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_split=hp["min_samples_split"],
                max_features=hp["max_features"],
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            return clf
        rf_choices = {
            "n_estimators": [200, 400, 800],
            "max_depth": [10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        }

    spaces["rf"] = dict(
        make_estimator=make_rf,
        hyper_choices=rf_choices,
    )

    # ----- ExtraTrees -----
    def make_extratrees(hp):
        clf = ExtraTreesClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"],
            max_features=hp["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return clf

    spaces["extratrees"] = dict(
        make_estimator=make_extratrees,
        hyper_choices={
            "n_estimators": [200, 400, 800, 1200],
            "max_depth": [10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        },
    )

    # ----- Gaussian NB -----
    def make_gnb(hp):
        return GaussianNB(var_smoothing=hp["var_smoothing"])

    spaces["gnb"] = dict(
        make_estimator=make_gnb,
        hyper_choices={
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
    )

    # ----- XGBoost -----
    if HAVE_XGB:
        tree_method = "hist"

        def make_xgb(hp):
            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method=tree_method,
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                subsample=hp["subsample"],
                colsample_bytree=hp["colsample_bytree"],
                reg_lambda=hp["reg_lambda"],
                n_jobs=1,
                random_state=RANDOM_STATE,
            )
            return clf

        spaces["xgb"] = dict(
            make_estimator=make_xgb,
            hyper_choices={
                "n_estimators": [300, 500, 800],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0],
            },
        )

    return spaces


def compute_balanced_sample_weights(y_enc: np.ndarray) -> np.ndarray:
    """
    Per-sample weights so that each class contributes equally to the loss.
    y_enc: 1-D array of encoded class labels (ints).
    Returns 1-D float32 array of weights.
    """
    y_enc = np.asarray(y_enc)
    classes = np.unique(y_enc)
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_enc,
    )
    class2w = {c: w for c, w in zip(classes, cw)}
    return np.asarray([class2w[c] for c in y_enc], dtype=np.float32)

def build_fit_params_with_sample_weight(estimator, sample_weight):
    """
    Build fit_params dict for Pipeline.fit so that 'sample_weight'
    is passed to the final 'clf' step *only if* its .fit() supports it.
    """
    fit_params = {}
    if sample_weight is None:
        return fit_params

    clf = None
    try:
        clf = estimator.named_steps.get("clf", None)
    except Exception:
        clf = None

    if clf is not None:
        sig = inspect.signature(clf.fit)
        if "sample_weight" in sig.parameters:
            fit_params["clf__sample_weight"] = sample_weight

    return fit_params


def _plan_random_trials(
    n_classes: int,
    outer_k: int,
    inner_k: int,
    max_fits: int,
    random_state: int = SUPERVISED_RANDOM_SEED,
):
    """
    Plan a *global* set of trial configs for nested CV.

    Each trial (hyperparameter combo) will be evaluated with:

        outer_k outer folds × inner_k inner folds

    We reserve at least `outer_k` extra fits for the final
    outer-fold training (one fit per outer fold with the
    hyperparams selected by inner CV).
    """
    random.seed(random_state)
    np.random.seed(random_state)

    bio_specs = _build_bio_specs()
    spaces = model_search_spaces(n_classes)
    model_names = list(spaces.keys())
    repr_options = ["hvg", "pca", "pca_whiten", "ae"]

    # Count tables for *balanced* sampling
    bio_counts = {b["tag"]: 0 for b in bio_specs}
    bio_by_tag = {b["tag"]: b for b in bio_specs}
    hvg_counts = {n: 0 for n in HVG_COUNTS}
    repr_counts = {rk: 0 for rk in repr_options}

    all_pcs = sorted({pc for n in HVG_COUNTS for pc in _pc_grid_for(n)})
    pc_counts = {pc: 0 for pc in all_pcs}

    ae_latent_choices = AE_LATENT_GRID
    ae_latent_counts = {d: 0 for d in ae_latent_choices}

    marker_flag_values = [False, True]
    marker_flag_counts = {v: 0 for v in marker_flag_values}

    model_counts = {m: 0 for m in model_names}
    hp_counts = {
        m: {
            hp: {val: 0 for val in spec["hyper_choices"][hp]}
            for hp in spec["hyper_choices"]
        }
        for m, spec in spaces.items()
    }

    # ---- budget logic for nested CV ----
    fits_per_trial = max(1, outer_k * inner_k)

    # Reserve at least one final fit per outer fold
    usable_fits = max_fits - outer_k
    if usable_fits <= 0:
        raise ValueError(
            f"SUPERVISED_MAX_FITS={max_fits} too small for "
            f"outer_k={outer_k}, inner_k={inner_k}. "
            f"Need at least outer_k extra fits."
        )

    max_possible_trials = len(HVG_COUNTS) * len(bio_specs) * len(model_names)
    n_trials = max(
        SUPERVISED_MIN_TRIALS,
        min(max_possible_trials, usable_fits // fits_per_trial),
    )
    if n_trials * fits_per_trial > usable_fits:
        n_trials = max(1, usable_fits // fits_per_trial)

    planned_fits = n_trials * fits_per_trial + outer_k
    log(
        f"Planning {n_trials} supervised trials for nested CV "
        f"({outer_k} outer × {inner_k} inner "
        f"→ ≤ {planned_fits} fits, budget={max_fits})."
    )

    trials = []
    for t in range(n_trials):
        bio_tag = _pick_with_min_counts(bio_counts, list(bio_counts.keys()))
        bio = bio_by_tag[bio_tag]

        hvg = _pick_with_min_counts(hvg_counts, HVG_COUNTS)

        repr_kind = _pick_with_min_counts(repr_counts, repr_options)
        repr_kind = str(repr_kind)

        if repr_kind in ("pca", "pca_whiten"):
            pc_opts = _pc_grid_for(hvg)
            n_pcs = _pick_with_min_counts(pc_counts, pc_opts)
            ae_latent_dim = 0
        elif repr_kind == "hvg":
            n_pcs = 0
            ae_latent_dim = 0
        elif repr_kind == "ae":
            n_pcs = 0
            ae_latent_dim = _pick_with_min_counts(
                ae_latent_counts,
                ae_latent_choices,
            )
        else:
            raise ValueError(f"Unknown repr_kind={repr_kind!r} in planner.")

        use_marker_scores = _pick_with_min_counts(
            marker_flag_counts, marker_flag_values
        )

        model = _pick_with_min_counts(model_counts, model_names)
        spec = spaces[model]

        hp_vals = {}
        for hp_name, options in spec["hyper_choices"].items():
            val = _pick_with_min_counts(hp_counts[model][hp_name], options)
            hp_vals[hp_name] = val

        bio_counts[bio_tag] += 1
        hvg_counts[hvg] += 1
        repr_counts[repr_kind] += 1
        if repr_kind in ("pca", "pca_whiten"):
            pc_counts[n_pcs] += 1
        if repr_kind == "ae":
            ae_latent_counts[ae_latent_dim] += 1
        model_counts[model] += 1
        for hp_name, val in hp_vals.items():
            hp_counts[model][hp_name][val] += 1

        marker_flag_counts[use_marker_scores] += 1

        trials.append(dict(
            trial_id=t + 1,
            combo=f"trial_{t+1:04d}",

            bio_tag=bio_tag,
            stage=bio["stage"],
            filter_level=bio["level"],
            regress_cc=bio["regress_cc"],

            n_top_hvg=hvg,
            repr_kind=repr_kind,
            n_pcs=n_pcs,
            ae_latent_dim=ae_latent_dim,
            use_marker_scores=bool(use_marker_scores),

            model=model,
            model_params=hp_vals,
        ))

    return trials, spaces



log(f"Using Torch device: {TORCH_DEVICE}")

# ---------------------- 9) Nested supervised RANDOM search (INNER_K × OUTER_K) ----------------------
from pathlib import Path as _P
OUT_DIR_P = _P(OUT_DIR)

log(f"Using Torch device: {TORCH_DEVICE}")

# ---------------------- Build meta_df and dummy X_full_df for supervised CV ----------------------
meta_df = pd.DataFrame({
    "cell_id": adata.obs["cell_id"].astype(str).values,
    "train_label": adata.obs["train_label"].values,
    "is_unlabeled_target": adata.obs["is_unlabeled_target"].values,
}).set_index("cell_id")

X_full_df = pd.DataFrame(
    {"_dummy": np.zeros(adata.n_obs, dtype=np.float32)},
    index=meta_df.index,
)

ids_all = X_full_df.index.values
y_all = meta_df.loc[ids_all, "train_label"].values

lab_mask = ~pd.isna(y_all)
X_lab_df = X_full_df.loc[ids_all[lab_mask]]
y_lab = y_all[lab_mask].astype(str)

le_global = LabelEncoder().fit(y_lab)
class_names_global = le_global.classes_
y_lab_enc = le_global.transform(y_lab)
n_classes = len(class_names_global)

trials, model_spaces = _plan_random_trials(
    n_classes=n_classes,
    outer_k=OUTER_K,
    inner_k=INNER_K,
    max_fits=SUPERVISED_MAX_FITS,
)

outer_cv = StratifiedKFold(
    n_splits=OUTER_K,
    shuffle=True,
    random_state=RANDOM_STATE,
)

all_rows = []
outer_rows = []
fit_counter = 0
outer_all_y_true = []
outer_all_y_pred = []


# For per-trial aggregated inner-CV stats used to pick final best combo)
trial_inner_stats = {
    t["combo"]: {
        "acc_means": [],
        "f1_means": [],
        "trial": t,
    }
    for t in trials
}

planned_fits = OUTER_K * INNER_K * len(trials) + OUTER_K
pbar_total = min(SUPERVISED_MAX_FITS, planned_fits)
pbar = tqdm(
    total=pbar_total,
    desc="Supervised fits (nested)",
    unit="fit",
    leave=False,
)

t_super = time.perf_counter()
for outer_fold, (outer_tr_idx, outer_te_idx) in enumerate(
    outer_cv.split(X_lab_df, y_lab_enc), 1
):
    log(
        f"[outer {outer_fold}/{OUTER_K}] "
        f"train={len(outer_tr_idx)}  test={len(outer_te_idx)}"
    )

    X_outer_tr = X_lab_df.iloc[outer_tr_idx]
    X_outer_te = X_lab_df.iloc[outer_te_idx]
    y_outer_tr = y_lab_enc[outer_tr_idx]
    y_outer_te = y_lab_enc[outer_te_idx]
    inner_cv = StratifiedKFold(
        n_splits=INNER_K,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )

    fold_trial_summaries = []

    for trial in trials:
        combo_key = trial["combo"]
        t_trial = time.perf_counter()
        log(
            f"[outer {outer_fold}] [trial {trial['trial_id']:04d}] {combo_key} | "
            f"stage={trial['stage']} level={trial['filter_level']} "
            f"regress_cc={trial['regress_cc']} HVG={trial['n_top_hvg']} "
            f"repr={trial['repr_kind']} n_pcs={trial['n_pcs']} "
            f"ae_latent={trial['ae_latent_dim']} model={trial['model']} "
            f"markers={trial['use_marker_scores']} "
            f"hp={trial['model_params']}"
        )

        hvgt_step = ("hvgpca", HVGPCATransformer(
            n_top_hvg=trial["n_top_hvg"],
            stage=trial["stage"],
            filter_level=trial["filter_level"],
            regress_cc=trial["regress_cc"],
            n_pcs=trial["n_pcs"],
            use_gpu_pca=GPU_PCA,
            repr_kind=trial["repr_kind"],
            ae_latent_dim=trial["ae_latent_dim"],
            ae_epochs=AE_EPOCHS,
            ae_batch_size=AE_BATCH_SIZE,
            use_marker_scores=trial["use_marker_scores"],
        ))

        spec = model_spaces[trial["model"]]
        base_est = spec["make_estimator"](trial["model_params"])

        if isinstance(base_est, Pipeline):
            steps = base_est.steps
            est = Pipeline([hvgt_step] + steps, memory=PIPELINE_CACHE)
        else:
            est = Pipeline([hvgt_step, ("clf", base_est)], memory=PIPELINE_CACHE)

        inner_acc = []
        inner_f1 = []

        for inner_fold, (tr_idx, va_idx) in enumerate(
            inner_cv.split(X_outer_tr, y_outer_tr), 1
        ):
            t_fold = time.perf_counter()
            Xtr = X_outer_tr.iloc[tr_idx]
            Xva = X_outer_tr.iloc[va_idx]
            ytr = y_outer_tr[tr_idx]
            yva = y_outer_tr[va_idx]

            # class-balanced weights on *inner* training split
            sw_tr = compute_balanced_sample_weights(ytr)
            fit_params = build_fit_params_with_sample_weight(est, sw_tr)

            est.fit(Xtr, ytr, **fit_params)
            pred = est.predict(Xva)

            acc = accuracy_score(yva, pred)
            f1 = f1_score(yva, pred, average="macro")

            inner_acc.append(acc)
            inner_f1.append(f1)

            fit_counter += 1
            if fit_counter <= pbar_total:
                pbar.update(1)

            all_rows.append({
                "combo": combo_key,
                "trial_id": trial["trial_id"],
                "variant_stage": trial["stage"],
                "filter_level": trial["filter_level"],
                "regress_cc": trial["regress_cc"],
                "hvg_n": trial["n_top_hvg"],
                "repr_kind": trial["repr_kind"],
                "ae_latent_dim": trial["ae_latent_dim"],
                "n_pcs": trial["n_pcs"],
                "use_marker_scores": trial["use_marker_scores"],
                "model": trial["model"],
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "acc": acc,
                "f1_macro": f1,
                "best_params": json.dumps(trial["model_params"]),
            })

            log(
                f"[outer {outer_fold}] [{combo_key}] "
                f"inner_fold={inner_fold}/{INNER_K} "
                f"→ acc={acc:.4f} f1={f1:.4f} "
                f"({_fmt_dur(time.perf_counter()-t_fold)})"
            )

        if not inner_acc:
            continue

        mean_acc = float(np.mean(inner_acc))
        std_acc = float(np.std(inner_acc))
        mean_f1 = float(np.mean(inner_f1))
        std_f1 = float(np.std(inner_f1))

        fold_trial_summaries.append({
            "combo": combo_key,
            "trial_id": trial["trial_id"],
            "variant_stage": trial["stage"],
            "filter_level": trial["filter_level"],
            "regress_cc": trial["regress_cc"],
            "hvg_n": trial["n_top_hvg"],
            "repr_kind": trial["repr_kind"],
            "ae_latent_dim": trial["ae_latent_dim"],
            "n_pcs": trial["n_pcs"],
            "use_marker_scores": trial["use_marker_scores"],
            "model": trial["model"],
            "outer_fold": outer_fold,
            "inner_acc_mean": mean_acc,
            "inner_acc_std": std_acc,
            "inner_f1_mean": mean_f1,
            "inner_f1_std": std_f1,
        })

        trial_inner_stats[combo_key]["acc_means"].append(mean_acc)
        trial_inner_stats[combo_key]["f1_means"].append(mean_f1)

        log(
            f"[outer {outer_fold}] [{combo_key}] "
            f"inner-CV mean acc={mean_acc:.4f} ±{std_acc:.4f}, "
            f"mean f1={mean_f1:.4f} ±{std_f1:.4f} "
            f"({_fmt_dur(time.perf_counter()-t_trial)})"
        )

    if not fold_trial_summaries:
        log(f"[outer {outer_fold}] WARNING: no fold_trial_summaries; stopping nested CV.")
        break

    # ---- Select best trial inside this outer fold based on *inner* mean F1 ----
    best_inner = max(fold_trial_summaries, key=lambda r: r["inner_f1_mean"])
    best_combo = best_inner["combo"]
    best_trial_id = int(best_inner["trial_id"])

    log(
        f"[outer {outer_fold}] Selected best trial by inner-CV: "
        f"{best_combo} (id={best_trial_id}) "
        f"inner_f1_mean={best_inner['inner_f1_mean']:.4f}"
    )

    # Recover corresponding trial spec
    trial_spec = None
    for t in trials:
        if t["combo"] == best_combo and t["trial_id"] == best_trial_id:
            trial_spec = t
            break
    if trial_spec is None:
        raise RuntimeError(
            f"Could not find trial spec for combo={best_combo}, "
            f"trial_id={best_trial_id}"
        )

    # ---- Outer-fold final model: fit on ALL outer-training cells with the chosen hyperparams ----
    hvgt_best = ("hvgpca", HVGPCATransformer(
        n_top_hvg=trial_spec["n_top_hvg"],
        stage=trial_spec["stage"],
        filter_level=trial_spec["filter_level"],
        regress_cc=trial_spec["regress_cc"],
        n_pcs=trial_spec["n_pcs"],
        use_gpu_pca=GPU_PCA,
        repr_kind=trial_spec["repr_kind"],
        ae_latent_dim=trial_spec["ae_latent_dim"],
        ae_epochs=AE_EPOCHS,
        ae_batch_size=AE_BATCH_SIZE,
        use_marker_scores=trial_spec.get("use_marker_scores", False),
    ))

    spec_best = model_spaces[trial_spec["model"]]
    base_best = spec_best["make_estimator"](trial_spec["model_params"])

    if isinstance(base_best, Pipeline):
        steps_best = base_best.steps
        est_outer = Pipeline([hvgt_best] + steps_best, memory=PIPELINE_CACHE)
    else:
        est_outer = Pipeline([hvgt_best, ("clf", base_best)], memory=PIPELINE_CACHE)

    sw_outer = compute_balanced_sample_weights(y_outer_tr)
    fit_params_outer = build_fit_params_with_sample_weight(est_outer, sw_outer)

    t_outer_fit = time.perf_counter()
    est_outer.fit(X_outer_tr, y_outer_tr, **fit_params_outer)
    fit_counter += 1
    if fit_counter <= pbar_total:
        pbar.update(1)

    pred_outer = est_outer.predict(X_outer_te)
    acc_outer = accuracy_score(y_outer_te, pred_outer)
    f1_outer = f1_score(y_outer_te, pred_outer, average="macro")
    outer_all_y_true.append(y_outer_te.copy())
    outer_all_y_pred.append(pred_outer.copy())

    outer_rows.append({
        "outer_fold": outer_fold,
        "combo": best_combo,
        "trial_id": best_trial_id,
        "variant_stage": trial_spec["stage"],
        "filter_level": trial_spec["filter_level"],
        "regress_cc": trial_spec["regress_cc"],
        "hvg_n": trial_spec["n_top_hvg"],
        "repr_kind": trial_spec["repr_kind"],
        "ae_latent_dim": trial_spec["ae_latent_dim"],
        "n_pcs": trial_spec["n_pcs"],
        "use_marker_scores": trial_spec.get("use_marker_scores", False),
        "model": trial_spec["model"],
        "acc": acc_outer,
        "f1_macro": f1_outer,
        "best_params": json.dumps(trial_spec["model_params"]),
    })

    log(
        f"[outer {outer_fold}] nested OUTER score: "
        f"acc={acc_outer:.4f} f1={f1_outer:.4f} "
        f"({_fmt_dur(time.perf_counter()-t_outer_fit)})"
    )

pbar.close()
log(
    f"Nested supervised random search finished in "
    f"{_fmt_dur(time.perf_counter()-t_super)} "
    f"with {fit_counter} fits over {len(trials)} trials."
)

if outer_all_y_true:
    y_true_nested = np.concatenate(outer_all_y_true)
    y_pred_nested = np.concatenate(outer_all_y_pred)

    cm_prefix = os.path.join(FIG_DIR, "confmat_nested_pipeline")
    plot_confusion_from_preds(
        y_true_nested,
        y_pred_nested,
        class_names_global,
        cm_prefix,
    )
    log(f"→ wrote {cm_prefix}.png (nested-CV confusion matrix)")
else:
    log("No outer-fold predictions collected; nested confusion matrix skipped.")


# ---- Save INNER-CV rows (used for hyperparam analysis & best combo selection) ----
if all_rows:
    cv_df = pd.DataFrame(all_rows)
    cv_path = OUT_DIR_P / "nested_cv_summary.csv"
    cv_df.to_csv(cv_path, index=False)
    log(f"→ wrote {cv_path} ({len(cv_df)} inner-CV rows)")
else:
    cv_df = pd.DataFrame()
    log("No supervised inner-CV rows collected — check configuration / budget.")

# ---- Save OUTER nested scores (true unbiased evaluation of the tuning procedure) ----
if outer_rows:
    outer_df = pd.DataFrame(outer_rows)
    outer_path = OUT_DIR_P / "nested_outer_scores.csv"
    outer_df.to_csv(outer_path, index=False)
    log(
        f"→ wrote {outer_path} ({len(outer_df)} outer-fold rows); "
        f"mean acc={outer_df['acc'].mean():.4f} ±{outer_df['acc'].std():.4f}, "
        f"mean f1={outer_df['f1_macro'].mean():.4f} ±{outer_df['f1_macro'].std():.4f}"
    )
else:
    log("No outer nested rows collected (unexpected)")

# ---- Aggregate per-trial stats from INNER-CV means (across outer folds) ----
best_rows = []
for combo, stats in trial_inner_stats.items():
    acc_means = stats["acc_means"]
    f1_means = stats["f1_means"]
    if not acc_means:
        continue
    trial = stats["trial"]
    acc_means = np.asarray(acc_means, dtype=float)
    f1_means = np.asarray(f1_means, dtype=float)

    best_rows.append({
        "combo": combo,
        "trial_id": trial["trial_id"],
        "variant_stage": trial["stage"],
        "filter_level": trial["filter_level"],
        "regress_cc": trial["regress_cc"],
        "hvg_n": trial["n_top_hvg"],
        "repr_kind": trial["repr_kind"],
        "ae_latent_dim": trial["ae_latent_dim"],
        "n_pcs": trial["n_pcs"],
        "model": trial["model"],
        "use_marker_scores": trial["use_marker_scores"],
        "cv_acc_mean": float(acc_means.mean()),
        "cv_acc_std": float(acc_means.std()),
        "cv_f1_mean": float(f1_means.mean()),
        "cv_f1_std": float(f1_means.std()),
        "avg_best_params": json.dumps(trial["model_params"]),
    })

if best_rows:
    best_df = pd.DataFrame(best_rows).sort_values(
        "cv_f1_mean", ascending=False
    )
else:
    best_df = pd.DataFrame(columns=[
        "combo", "trial_id", "variant_stage", "filter_level",
        "regress_cc", "hvg_n", "repr_kind", "ae_latent_dim",
        "n_pcs", "model", "use_marker_scores",
        "cv_acc_mean", "cv_acc_std", "cv_f1_mean", "cv_f1_std",
        "avg_best_params",
    ])

best_out = OUT_DIR_P / "best_model_per_combo.csv"
best_df.to_csv(best_out, index=False)
print("\n[supervised] ✅ Nested random-search CV (inner level) complete.")
print(best_df.head(15).to_string(index=False))
log(f"→ wrote {best_out}")


# ---------------------- 10) FINAL NUMERIC SUBMISSION (refit BEST trial once) ----------------------
if len(best_df) == 0:
    raise RuntimeError(
        "No supervised trials completed — check earlier steps and budget."
    )

# Pick best row by cv_f1_mean
best_row = best_df.iloc[0].copy()
best_combo = best_row["combo"]
best_model_name = best_row["model"]
best_trial_id = int(best_row.get("trial_id", 1))

log(
    f"Best trial: {best_combo} (id={best_trial_id}) "
    f"model={best_model_name} repr={best_row['repr_kind']} "
    f"hvg={best_row['hvg_n']} pcs={best_row['n_pcs']} "
    f"ae_latent={best_row['ae_latent_dim']} "
    f"f1={best_row['cv_f1_mean']:.4f}"
)


# Recover the corresponding trial spec (stage/filter/HPs)
trial_lookup = {
    (t["combo"], t["model"], t["trial_id"]): t
    for t in trials
}
best_key = (best_combo, best_model_name, best_trial_id)
if best_key not in trial_lookup:
    # fallback: try ignoring trial_id
    for t in trials:
        if t["combo"] == best_combo and t["model"] == best_model_name:
            trial_spec = t
            break
    else:
        raise RuntimeError("Could not find best trial spec in trials list.")
else:
    trial_spec = trial_lookup[best_key]

log(f"Refitting best trial spec: {trial_spec}")

# Build final estimator and fit on ALL labeled data
hvgt_best = ("hvgpca", HVGPCATransformer(
    n_top_hvg=trial_spec["n_top_hvg"],
    stage=trial_spec["stage"],
    filter_level=trial_spec["filter_level"],
    regress_cc=trial_spec["regress_cc"],
    n_pcs=trial_spec["n_pcs"],
    use_gpu_pca=GPU_PCA,
    repr_kind=trial_spec["repr_kind"],
    ae_latent_dim=trial_spec["ae_latent_dim"],
    ae_epochs=AE_EPOCHS,
    ae_batch_size=AE_BATCH_SIZE,
    use_marker_scores=trial_spec.get("use_marker_scores", False),
))

spec_best = model_spaces[best_model_name]
base_best = spec_best["make_estimator"](trial_spec["model_params"])

if isinstance(base_best, Pipeline):
    steps_best = base_best.steps
    final_est = Pipeline([hvgt_best] + steps_best, memory=PIPELINE_CACHE)
else:
    final_est = Pipeline([hvgt_best, ("clf", base_best)], memory=PIPELINE_CACHE)

t_refit = time.perf_counter()
sw_all = compute_balanced_sample_weights(y_lab_enc)
fit_params_best = build_fit_params_with_sample_weight(final_est, sw_all)
final_est.fit(X_lab_df, y_lab_enc, **fit_params_best)
log(f"Final refit of best estimator: {_fmt_dur(time.perf_counter()-t_refit)}")


tag = os.path.join(FIG_DIR, f"diag_{best_combo}__{best_model_name}")
plot_learning_curves_mlp(final_est, tag)

# Permutation importance
try:
    n_cols = X_lab_df.shape[1]
    clf_step = final_est.named_steps.get("clf", None)
    if n_cols <= 800 and not isinstance(clf_step, TorchMLPClassifier):
        r = permutation_importance(
            final_est,
            X_lab_df,
            y_lab_enc,
            n_repeats=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scoring="f1_macro",
        )
        idx = np.argsort(r.importances_mean)[-20:]
        plt.figure(figsize=(7, 5))
        plt.barh(range(len(idx)), r.importances_mean[idx])
        plt.yticks(
            range(len(idx)),
            [f"{i}" for i in idx]
        )
        plt.title("Permutation importance (top 20)")
        plt.tight_layout()
        plt.savefig(f"{tag}_perm_importance.png", dpi=150)
        plt.close()
except Exception:
    pass


# Predict unlabeled
unlab_mask = meta_df.loc[ids_all, "is_unlabeled_target"].values.astype(bool)
X_unlab_df = X_full_df.loc[ids_all[unlab_mask]]
if hasattr(final_est, "predict_proba"):
    proba_unlab = final_est.predict_proba(X_unlab_df)
else:
    preds_tmp = final_est.predict(X_unlab_df)
    proba_unlab = np.zeros((len(preds_tmp), len(class_names_global)))
    proba_unlab[np.arange(len(preds_tmp)), preds_tmp] = 1.0

if hasattr(
    final_est.named_steps.get("clf", final_est),
    "classes_"
):
    numeric_order = (
        final_est.named_steps.get("clf", final_est)
        .classes_
    )
    class_order = le_global.inverse_transform(numeric_order)
else:
    class_order = class_names_global

pred_unlab_num = np.argmax(proba_unlab, axis=1)
pred_unlab_lbl = class_order[pred_unlab_num]

out = pd.DataFrame({
    "ID": X_unlab_df.index.values,
    "pred_cell_type": pred_unlab_lbl,
    "pred_confidence": proba_unlab.max(axis=1),
})
for j, cname in enumerate(class_order):
    out[f"prob_{cname}"] = proba_unlab[:, j]

pred_path = OUT_DIR_P / f"preds_{best_combo}__{best_model_name}.csv"
out.to_csv(pred_path, index=False)
print(f"→ wrote predictions: {pred_path}")

# Keep a tiny best_df with this refit info & preds path
best_row = best_row.copy()
best_row["best_model"] = best_model_name
best_row["refit_best_params"] = json.dumps(trial_spec["model_params"])
best_row["preds_csv"] = str(pred_path)
best_df_final = pd.DataFrame([best_row])
best_df_final.to_csv(OUT_DIR_P / "best_model_final.csv", index=False)
log(f"→ wrote {OUT_DIR_P / 'best_model_final.csv'}")

# ---------------------- 11) FINAL submission file ----------------------
if pred_path is None or not os.path.exists(pred_path):
    raise RuntimeError("Prediction CSV for best model not found.")

preds_df = pd.read_csv(pred_path)

sorted_classes = sorted(
    preds_df.filter(like="prob_")
    .columns.str.replace("prob_", "", regex=False)
)
name_to_num = {name: i for i, name in enumerate(sorted_classes)}
submission = preds_df[["ID", "pred_cell_type"]].copy()
submission.rename(columns={"pred_cell_type": "clusters"}, inplace=True)
submission["clusters"] = submission["clusters"].map(
    name_to_num
).astype("Int64")

final_submission_path = os.path.join(SUBMISSION_DIR, FINAL_SUBMISSION_CSV)
submission.to_csv(final_submission_path, index=False)
print(f"\n✅ FINAL submission written to: {final_submission_path}")
print(submission.head())

# ---------------------- 11b) Extra submissions: per-model + ensembles ----------------------

def get_full_proba(est, X_df, n_classes):
    """
    Return a (n_samples × n_classes) probability matrix in the *global*
    label-encoding order (0..n_classes-1).
    """
    n_samples = X_df.shape[0]
    proba_full = np.zeros((n_samples, n_classes), dtype=float)

    clf = est.named_steps.get("clf", est) if hasattr(est, "named_steps") else est

    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_df)
        if hasattr(clf, "classes_"):
            classes = np.asarray(clf.classes_, dtype=int)
            for j, g in enumerate(classes):
                if 0 <= g < n_classes:
                    proba_full[:, g] = proba[:, j]
        else:
            n_local = proba.shape[1]
            proba_full[:, :n_local] = proba
    else:
        preds = est.predict(X_df)
        for i, g in enumerate(preds):
            g_int = int(g)
            if 0 <= g_int < n_classes:
                proba_full[i, g_int] = 1.0
    return proba_full



def write_preds_and_submission(tag, proba_unlab, sample_ids, class_names_global):
    """
    Helper to write:
      - OUT_DIR/preds_{tag}.csv with per-class probabilities
      - SUBMISSION_DIR/submission_{tag}.csv with 'clusters' ints
    """
    sample_ids = np.asarray(sample_ids, dtype=str)
    n_samples, n_classes = proba_unlab.shape
    assert n_classes == len(class_names_global)

    pred_num = proba_unlab.argmax(axis=1)
    pred_lbl = class_names_global[pred_num]
    preds_df = pd.DataFrame({
        "ID": sample_ids,
        "pred_cell_type": pred_lbl,
        "pred_confidence": proba_unlab.max(axis=1),
    })
    for j, cname in enumerate(class_names_global):
        preds_df[f"prob_{cname}"] = proba_unlab[:, j]

    pred_path = OUT_DIR_P / f"preds_{tag}.csv"
    preds_df.to_csv(pred_path, index=False)

    # Map class names to numeric cluster ids (sorted lexicographically)
    sorted_classes = sorted(class_names_global)
    name_to_num = {name: i for i, name in enumerate(sorted_classes)}
    submission = preds_df[["ID", "pred_cell_type"]].copy()
    submission.rename(columns={"pred_cell_type": "clusters"}, inplace=True)
    submission["clusters"] = submission["clusters"].map(name_to_num).astype("Int64")

    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{tag}.csv")
    submission.to_csv(sub_path, index=False)

    log(f"[extra] wrote predictions → {pred_path}")
    log(f"[extra] wrote submission → {sub_path}")
    return pred_path, sub_path


log("Building extra submissions: top-2 per model family and top-5/top-10 ensembles")

n_classes_global = len(class_names_global)

trial_lookup = {
    (t["combo"], t["model"], t["trial_id"]): t
    for t in trials
}

def _get_trial_spec(combo, model_name, trial_id):
    key = (combo, model_name, trial_id)
    if key in trial_lookup:
        return trial_lookup[key]
    for t in trials:
        if t["combo"] == combo and t["model"] == model_name:
            return t
    raise RuntimeError(
        f"Could not find trial spec for combo={combo}, model={model_name}, trial_id={trial_id}"
    )

# Collect all trial rows we need to refit
selected_specs = {}

# 1) Best overall trial
key_best = (best_combo, best_model_name, best_trial_id)
selected_specs[key_best] = {
    "row": best_row,
    "tags": {"best_overall"},
    "family_model": best_model_name,
    "family_ranks": [1],
}

# 2) Top-2 trials per model family
for model_name in sorted(best_df["model"].unique()):
    sub = best_df[best_df["model"] == model_name]
    if sub.empty:
        continue
    for rank, (_, row_m) in enumerate(sub.head(2).iterrows(), 1):
        key = (row_m["combo"], row_m["model"], int(row_m.get("trial_id", 1)))
        entry = selected_specs.setdefault(
            key,
            {
                "row": row_m,
                "tags": set(),
                "family_model": model_name,
                "family_ranks": [],
            },
        )
        entry["tags"].add("per_family")
        if rank not in entry["family_ranks"]:
            entry["family_ranks"].append(rank)

# 3) Overall top-5 and top-10 trials
for _, row_k in best_df.head(5).iterrows():
    key = (row_k["combo"], row_k["model"], int(row_k.get("trial_id", 1)))
    entry = selected_specs.setdefault(
        key,
        {
            "row": row_k,
            "tags": set(),
            "family_model": row_k["model"],
            "family_ranks": [],
        },
    )
    entry["tags"].add("top5")

for _, row_k in best_df.head(10).iterrows():
    key = (row_k["combo"], row_k["model"], int(row_k.get("trial_id", 1)))
    entry = selected_specs.setdefault(
        key,
        {
            "row": row_k,
            "tags": set(),
            "family_model": row_k["model"],
            "family_ranks": [],
        },
    )
    entry["tags"].add("top10")

log(f"[extra] Selected {len(selected_specs)} distinct trial specs for refit")

# Refit all selected trials on *all* labeled data
model_results = {}

# Reuse already-fitted best model where possible
if key_best in selected_specs:
    proba_lab_best = get_full_proba(final_est, X_lab_df, n_classes_global)
    proba_unlab_best = get_full_proba(final_est, X_unlab_df, n_classes_global)
    info_best = selected_specs[key_best]
    model_results[key_best] = {
        "combo": best_combo,
        "model": best_model_name,
        "trial_id": best_trial_id,
        "est": final_est,
        "proba_lab": proba_lab_best,
        "proba_unlab": proba_unlab_best,
        "tags": info_best["tags"],
        "row": info_best["row"],
        "family_model": info_best["family_model"],
        "family_ranks": info_best["family_ranks"],
    }

for (combo, model_name, trial_id), info in selected_specs.items():
    if (combo, model_name, trial_id) == key_best:
        continue

    trial_spec = _get_trial_spec(combo, model_name, int(trial_id))

    hvgt_step = ("hvgpca", HVGPCATransformer(
        n_top_hvg=trial_spec["n_top_hvg"],
        stage=trial_spec["stage"],
        filter_level=trial_spec["filter_level"],
        regress_cc=trial_spec["regress_cc"],
        n_pcs=trial_spec["n_pcs"],
        use_gpu_pca=GPU_PCA,
        repr_kind=trial_spec["repr_kind"],
        ae_latent_dim=trial_spec["ae_latent_dim"],
        ae_epochs=AE_EPOCHS,
        ae_batch_size=AE_BATCH_SIZE,
        use_marker_scores=trial_spec.get("use_marker_scores", False),
    ))

    spec_model = model_spaces[model_name]
    base_est = spec_model["make_estimator"](trial_spec["model_params"])
    if isinstance(base_est, Pipeline):
        est = Pipeline([hvgt_step] + base_est.steps, memory=PIPELINE_CACHE)
    else:
        est = Pipeline([hvgt_step, ("clf", base_est)], memory=PIPELINE_CACHE)

    log(
        f"[extra] Refitting combo={combo}, model={model_name}, "
        f"trial_id={trial_id} for extra outputs..."
    )
    t_refit = time.perf_counter()
    sw_all = compute_balanced_sample_weights(y_lab_enc)
    fit_params = build_fit_params_with_sample_weight(est, sw_all)
    est.fit(X_lab_df, y_lab_enc, **fit_params)
    log(f"[extra]   done in {_fmt_dur(time.perf_counter() - t_refit)}")

    proba_lab = get_full_proba(est, X_lab_df, n_classes_global)
    proba_unlab = get_full_proba(est, X_unlab_df, n_classes_global)

    model_results[(combo, model_name, trial_id)] = {
        "combo": combo,
        "model": model_name,
        "trial_id": int(trial_id),
        "est": est,
        "proba_lab": proba_lab,
        "proba_unlab": proba_unlab,
        "tags": info["tags"],
        "row": info["row"],
        "family_model": info.get("family_model", model_name),
        "family_ranks": info.get("family_ranks", []),
    }

# ---- Per-family top-2 submissions + confusion matrices ----
cv_cache = {}  # per non rifare la CV se lo stesso trial compare più volte
for key, res in model_results.items():
    if "per_family" not in res["tags"]:
        continue
    family_model = res["family_model"]
    ranks = res.get("family_ranks", []) or [1]

    trial_key = (res["combo"], res["model"], int(res["trial_id"]))
    if trial_key not in cv_cache:
        trial_spec = _get_trial_spec(res["combo"], res["model"], int(res["trial_id"]))
        y_true_cv, y_pred_cv = cv_predictions_for_trial(
            trial_spec,
            model_spaces,
            X_lab_df,
            y_lab_enc,
            outer_k=OUTER_K,
            random_state=RANDOM_STATE,
        )
        cv_cache[trial_key] = (y_true_cv, y_pred_cv)
    else:
        y_true_cv, y_pred_cv = cv_cache[trial_key]

    for rk in ranks:
        tag = f"{family_model}_top{rk}_trial{res['trial_id']}"
        cm_prefix = os.path.join(FIG_DIR, f"confmat_cv_{tag}")
        plot_confusion_from_preds(y_true_cv, y_pred_cv, class_names_global, cm_prefix)

        write_preds_and_submission(
            tag,
            res["proba_unlab"],
            X_unlab_df.index.values,
            class_names_global,
        )


# ---- Ensembles: overall top-5 and top-10 ----
top5_models = [
    res for res in model_results.values()
    if "top5" in res["tags"]
]
top10_models = [
    res for res in model_results.values()
    if "top10" in res["tags"]
]

# Sort by CV F1
top5_models = sorted(
    top5_models,
    key=lambda r: float(r["row"]["cv_f1_mean"]),
    reverse=True,
)[:5]
top10_models = sorted(
    top10_models,
    key=lambda r: float(r["row"]["cv_f1_mean"]),
    reverse=True,
)[:10]

def _build_ensemble_and_outputs(models, tag):
    if not models:
        log(f"[ensemble {tag}] skipped: no models available")
        return
    log(f"[ensemble {tag}] averaging over {len(models)} base models")
    proba_lab_stack = np.stack([m["proba_lab"] for m in models], axis=0)
    proba_unlab_stack = np.stack([m["proba_unlab"] for m in models], axis=0)
    proba_lab_ens = proba_lab_stack.mean(axis=0)
    proba_unlab_ens = proba_unlab_stack.mean(axis=0)

    y_pred_lab = proba_lab_ens.argmax(axis=1)
    acc = accuracy_score(y_lab_enc, y_pred_lab)
    f1 = f1_score(y_lab_enc, y_pred_lab, average="macro")
    log(f"[ensemble {tag}] acc={acc:.4f}, f1_macro={f1:.4f}")
    ensemble_perf_records.append({
        "tag": tag,
        "n_models": len(models),
        "acc": float(acc),
        "f1_macro": float(f1),
    })

    cm_prefix = os.path.join(FIG_DIR, f"confmat_ensemble_{tag}")
    plot_confusion_from_preds(y_lab_enc, y_pred_lab, class_names_global, cm_prefix)

    write_preds_and_submission(f"ensemble_{tag}", proba_unlab_ens, X_unlab_df.index.values, class_names_global)

ensemble_perf_records = []
_build_ensemble_and_outputs(top5_models, "top5")
_build_ensemble_and_outputs(top10_models, "top10")

log("Extra submissions & ensemble predictions complete.")

# ---------------------- 12) RUN METADATA + PERFORMANCE Visualization ----------------------
try:
    run_meta = {
        "n_combos": int(len(trials)) if "trials" in globals() else 0,
        "outer_k": int(OUTER_K),
        "inner_k": int(INNER_K),
        "timestamp": _ts(),
        "debug_mode": bool(DEBUG_MODE),
        "use_gpu": bool(USE_GPU),
        "torch_device": str(TORCH_DEVICE),
    }
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    log(f"Saved run metadata → {os.path.join(OUT_DIR, 'run_meta.json')}")
except Exception as e:
    log(f"[meta] skipped: {e}")

try:
    cv = pd.read_csv(OUT_DIR_P / "nested_cv_summary.csv")

    cv_agg = cv.groupby(
        [
            "combo",
            "model",
            "repr_kind",
            "variant_stage",
            "filter_level",
            "regress_cc",
            "hvg_n",
            "n_pcs",
        ],
        dropna=False,
    )["f1_macro"].mean().reset_index()

    # ---------- 1) F1 vs HVG count by model (line plot) ----------
    agg_hvg_model = (
        cv_agg.groupby(["model", "hvg_n"], dropna=False)["f1_macro"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    for m in sorted(agg_hvg_model["model"].unique()):
        sub = agg_hvg_model[agg_hvg_model["model"] == m].copy()
        sub = sub.sort_values("hvg_n")
        plt.plot(
            sub["hvg_n"],
            sub["f1_macro"],
            marker="o",
            label=m,
        )
    plt.xlabel("HVG count")
    plt.ylabel("mean F1 (inner-CV)")
    plt.title("F1 vs HVG count by model")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/perf_f1_vs_hvg_by_model.png", dpi=150)
    plt.close()

    # ---------- 2) Torch MLP: F1 vs #PCs across HVGs (line per HVG) ----------
    mlp = cv_agg[cv_agg["model"] == "mlp"].copy()
    if not mlp.empty:
        mlp_mean = (
            mlp.groupby(["hvg_n", "n_pcs"], dropna=False)["f1_macro"]
            .mean()
            .reset_index()
        )
        plt.figure(figsize=(8, 5))
        for n in sorted(mlp_mean["hvg_n"].unique()):
            sub = mlp_mean[mlp_mean["hvg_n"] == n].copy()
            sub = sub.sort_values("n_pcs")
            plt.plot(
                sub["n_pcs"],
                sub["f1_macro"],
                marker="o",
                label=f"HVG={n}",
            )
        plt.xlabel("#PCs (0 = HVG-only)")
        plt.ylabel("mean F1 (inner-CV)")
        plt.title("Torch MLP: F1 vs #PCs across HVGs")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(
            f"{FIG_DIR}/perf_mlp_f1_vs_pcs_across_hvg.png",
            dpi=150,
        )
        plt.close()

    # ---------- 3) Box: distribution by model ----------
    plt.figure(figsize=(8, 5))
    models = sorted(cv_agg["model"].unique())
    data = [cv_agg[cv_agg["model"] == m]["f1_macro"].values for m in models]
    bp = plt.boxplot(data, tick_labels=models, showmeans=True)
    add_counts_to_boxplot(plt.gca(), data, models)
    plt.ylabel("mean F1 (inner-CV)")
    plt.title("Distribution of F1 by model (across combos)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/perf_box_by_model.png", dpi=150)
    plt.close()

    # ---------- 4) Box: F1 by biology filter_level, per model ----------
    agg_full = cv_agg

    models = sorted(agg_full["model"].unique())
    for m in models:
        sub = agg_full[agg_full["model"] == m]
        levels = sorted(sub["filter_level"].dropna().unique())
        if not levels:
            continue
        data = [sub[sub["filter_level"] == lvl]["f1_macro"].values for lvl in levels]
        if all(len(d) == 0 for d in data):
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, tick_labels=levels, showmeans=True)
        add_counts_to_boxplot(ax, data, levels)
        ax.set_ylabel("mean F1 (inner-CV)")
        ax.set_xlabel("filter_level")
        ax.set_title(f"{m}: F1 by biology filter level")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/perf_f1_by_filter_level_{m}.png", dpi=150)
        plt.close(fig)

    # ---------- 5) Box: F1 by variant_stage (pre/post), per model ----------
    for m in models:
        sub = agg_full[agg_full["model"] == m]
        stages = sorted(sub["variant_stage"].dropna().unique())
        if not stages:
            continue
        data = [sub[sub["variant_stage"] == st]["f1_macro"].values for st in stages]
        if all(len(d) == 0 for d in data):
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, tick_labels=stages, showmeans=True)
        add_counts_to_boxplot(ax, data, stages)
        ax.set_ylabel("mean F1 (inner-CV)")
        ax.set_xlabel("variant_stage")
        ax.set_title(f"{m}: F1 by filter stage")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/perf_f1_by_stage_{m}.png", dpi=150)
        plt.close(fig)

    # ---------- 6) Box: F1 with/without cell-cycle regression ----------
    agg_full["regress_cc"] = agg_full["regress_cc"].astype(str)
    for m in models:
        sub = agg_full[agg_full["model"] == m]
        flags = sorted(sub["regress_cc"].dropna().unique())
        if not flags:
            continue
        data = [sub[sub["regress_cc"] == flag]["f1_macro"].values for flag in flags]
        if all(len(d) == 0 for d in data):
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, tick_labels=flags, showmeans=True)
        add_counts_to_boxplot(ax, data, flags)
        ax.set_ylabel("mean F1 (inner-CV)")
        ax.set_xlabel("regress_cc")
        ax.set_title(f"{m}: F1 with/without cell-cycle regression")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/perf_f1_by_ccreg_{m}.png", dpi=150)
        plt.close(fig)

    # ---------- 7) F1 vs HVG count by filter_level (line plot) ----------
    for m in models:
        sub = agg_full[agg_full["model"] == m]
        if sub.empty:
            continue
        levels = sorted(sub["filter_level"].dropna().unique())
        if not levels:
            continue

        plt.figure(figsize=(8, 5))
        for lvl in levels:
            s2 = (
                sub[sub["filter_level"] == lvl]
                .groupby("hvg_n", dropna=False)["f1_macro"]
                .mean()
                .reset_index()
            )
            if s2.empty:
                continue
            s2 = s2.sort_values("hvg_n")
            plt.plot(
                s2["hvg_n"],
                s2["f1_macro"],
                marker="o",
                label=f"filter={lvl}",
            )
        plt.xlabel("HVG count")
        plt.ylabel("mean F1 (inner-CV)")
        plt.title(f"{m}: F1 vs HVG count by filter_level")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/perf_f1_vs_hvg_by_filter_{m}.png", dpi=150)
        plt.close()

    # ---------- 8) F1 vs #PCs by filter_level (line plot) ----------
    for m in models:
        sub = agg_full[agg_full["model"] == m]
        if sub.empty or sub["n_pcs"].isna().all():
            continue
        levels = sorted(sub["filter_level"].dropna().unique())
        if not levels:
            continue

        plt.figure(figsize=(8, 5))
        for lvl in levels:
            s2 = (
                sub[sub["filter_level"] == lvl]
                .groupby("n_pcs", dropna=False)["f1_macro"]
                .mean()
                .reset_index()
            )
            if s2.empty:
                continue
            s2 = s2.sort_values("n_pcs")
            plt.plot(
                s2["n_pcs"],
                s2["f1_macro"],
                marker="o",
                label=f"filter={lvl}",
            )
        plt.xlabel("#PCs (0 = HVG-only)")
        plt.ylabel("mean F1 (inner-CV)")
        plt.title(f"{m}: F1 vs #PCs by filter_level")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/perf_f1_vs_pcs_by_filter_{m}.png", dpi=150)
        plt.close()

    # ---------- 9) Heatmap HVG × #PCs ----------
    for m in models:
        sub = agg_full[agg_full["model"] == m]
        if sub.empty:
            continue
        pivot = (
            sub.groupby(["hvg_n", "n_pcs"], dropna=False)["f1_macro"]
            .mean()
            .unstack("n_pcs")
        )
        if pivot.empty:
            continue

        hvg_vals = list(pivot.index)
        pc_vals = list(pivot.columns)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.colorbar(im, label="mean F1 (inner-CV)")
        plt.xticks(np.arange(len(pc_vals)), pc_vals)
        plt.yticks(np.arange(len(hvg_vals)), hvg_vals)
        plt.xlabel("#PCs")
        plt.ylabel("HVG count")
        plt.title(f"{m}: mean F1 heatmap (HVG × #PCs)")
        plt.tight_layout()
        plt.savefig(
            f"{FIG_DIR}/perf_f1_heatmap_hvg_vs_pcs_{m}.png",
            dpi=150,
        )
        plt.close()

except Exception as e:
    print(f"[perf plots] skipped: {e}")

# ---------------------- 13) META FEATURE IMPORTANCE  ----------------------
try:
    cv_path = OUT_DIR_P / "nested_cv_summary.csv"
    if not cv_path.exists():
        log("[meta-importance] nested_cv_summary.csv not found, skip.")
    else:
        log("[meta-importance] Loading nested_cv_summary.csv for hyperparameters analysis...")
        cv_meta = pd.read_csv(cv_path)

        param_dicts = cv_meta["best_params"].apply(json.loads)
        params_df = pd.json_normalize(param_dicts)

        for c in params_df.columns:
            params_df[c] = params_df[c].astype(str)

        cv_hp = pd.concat(
            [cv_meta.drop(columns=["best_params"]), params_df],
            axis=1
        )

        base_drop = [
            "combo",
            "trial_id",
            "outer_fold",
            "inner_fold",
            "acc",
        ]
        base_drop = [c for c in base_drop if c in cv_hp.columns]

        os.makedirs(FIG_DIR, exist_ok=True)

        for model_name in sorted(cv_hp["model"].unique()):
            sub = cv_hp[cv_hp["model"] == model_name].copy()
            if sub.empty:
                continue

            sub = sub.drop(columns=base_drop)

            y = sub["f1_macro"].values
            X = sub.drop(columns=["f1_macro"])

            X_enc = pd.get_dummies(X, drop_first=True)

            if X_enc.shape[1] == 0:
                log(f"[meta-importance] model {model_name}: no features after one-hot, skip.")
                continue

            log(f"[meta-importance] modello {model_name}: "
                f"{X_enc.shape[0]} rows, {X_enc.shape[1]} feature for RandomForestRegressor.")

            rf = RandomForestRegressor(
                n_estimators=400,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_enc, y)

            importances = pd.Series(
                rf.feature_importances_,
                index=X_enc.columns,
            ).sort_values(ascending=False)

            imp_csv = OUT_DIR_P / f"meta_hp_importance_{model_name}.csv"
            importances.to_csv(imp_csv, header=["importance"])
            log(f"[meta-importance] wrote {imp_csv} (top 10):")
            log(importances.head(10).to_string())

            top_k = min(20, len(importances))
            top_imp = importances.head(top_k)

            plt.figure(figsize=(9, 5))
            plt.barh(range(len(top_imp)), top_imp.values)
            plt.yticks(range(len(top_imp)), top_imp.index)
            plt.gca().invert_yaxis()
            plt.xlabel("importance (RandomForestRegressor on F1_macro)")
            plt.title(f"Meta feature importance parameters/hyper. – model: {model_name}")
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/meta_hp_importance_{model_name}.png", dpi=150)
            plt.close()

except Exception as e:
    log(f"[meta-importance] skipped: {e}")
