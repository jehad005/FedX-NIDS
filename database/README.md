# FedX-NIDS Post-Training & Metadata Database

This dedicated directory stores all **post-training outputs, evaluation metrics, learning history, and model registry records**.

---

## 🏗️ Data Architecture Separation

| Directory | Purpose | Contents |
| :--- | :--- | :--- |
| **`data/`** | **[INPUT ONLY]** Pre-training datasets | Original raw traffic CSVs (CIC-IDS2017, RT-IoT2022) & client non-IID partitions. |
| **`database/`** | **[OUTPUT ONLY]** Post-training results & metadata | SQLite database (`securefed_nids.db`) tracking model releases, training history, and plot references. |
| **`reports/`** | **[GRAPHS ONLY]** Visual analytics | Evaluation figures (Confusion Matrices, ROC curves, and XAI feature attributions). |

---

## 🗄️ Database Schema Overview (`securefed_nids.db`)

### 1. `model_versions`
Stores metadata for published baseline Deep Learning models (`v1.x`) and Federated Global Models (`v2.x`).
- `version`: Semver version string (e.g. `2.0.2`)
- `model_name`: Model display name (e.g. `FL Global Model`)
- `architecture`: Model architecture (e.g. `Top Fadarate Learning`, `EnsembleTop2`)
- `fl_method`: Winning aggregation algorithm (`fedprox`, `scaffold`, `krum`, etc.)
- `accuracy`, `notes`, `created_at`, `is_latest`

### 2. `training_history` (New)
Captures epoch-by-epoch loss and accuracy metrics during candidate and ensemble training.
- `version`, `epoch`, `train_loss`, `val_loss`, `train_acc`, `val_acc`, `learning_rate`, `recorded_at`

### 3. `evaluation_plots` (New)
References all post-training diagnostic figures and charts generated for auditing.
- `version`, `plot_type` (`confusion_matrix`, `roc_curve`, `learning_curve`), `file_path`, `title`, `created_at`

### 4. `fl_leaderboard` (New)
Records candidate benchmark evaluations during Auto-Best federated aggregation rounds.
- `round_idx`, `base_version`, `resulting_version`, `method`, `accuracy`, `f1`, `auc`, `is_winner`

### 5. `clients` & `client_updates`
Tracks registered distributed edge nodes and incoming differentially private (DP-SGD) weight update deltas.
- `client_id`, `round`, `samples`, `dp_clipping_norm`, `dp_noise_multiplier`, `status`

---

## 💾 Backup & Portability
- To back up the entire database, simply copy `database/securefed_nids.db`.
- The full SQL schema definition is maintained in [`database/schema.sql`](schema.sql).
