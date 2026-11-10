# FedX-NIDS (SecureFed-NIDS): Privacy-Preserving Federated Intrusion Detection System

**FedX-NIDS** is an end-to-end, privacy-preserving Network Intrusion Detection System (NIDS) combining **Ensemble Deep Learning**, **Federated Learning (FL)**, **Differential Privacy (DP-SGD)**, and **Dual-Layer Explainable AI (XAI)** across a Central HQ Console and distributed Edge Client Nodes.

---

## Key Capabilities

1. **6 Deep Learning Candidate Architectures:**
   - Multi-Layer Perceptron (`MLP` with BatchNorm & Dropout)
   - 1D-Convolutional Neural Network (`CNN` with dynamic sequence sizing)
   - Long Short-Term Memory (`LSTM`) & Bidirectional LSTM (`BiLSTM`)
   - Gated Recurrent Unit (`GRU`)
   - Transformer Encoder (`Self-Attention`)

2. **Stratified 5-Fold Cross-Validation & Statistical Rigor:**
   - Metrics: Accuracy, Precision, Recall, F1-Score, MCC, Specificity, and AUC-ROC.
   - Hypothesis Testing: Paired Student's t-test and McNemar's Test.

3. **Ensemble Modeling:**
   - Automatic Top-2 selection with Soft/Hard Voting, Weighted Average, or a Stacking Meta-Learner.

4. **Federated Learning & Byzantine-Robust Aggregation:**
   - Supports **FedAvg**, **FedProx**, **SCAFFOLD**, **FedNova**, and **MOON**.
   - Server Aggregator defends against poisoning via **Coordinate-wise Trimmed Mean**, **Coordinate-wise Median**, and **Krum**.

5. **Differential Privacy (DP-SGD):**
   - Gradient $L_2$-norm clipping and Gaussian noise perturbation with $(\varepsilon, \delta)$ budget tracking.
   - Edge nodes only upload protected parameter deltas $(\Delta W)$; raw packet flows stay local.

6. **Dual-Layer Explainable AI (XAI):**
   - **Central HQ Audit (`/xai`):** PyTorch gradient feature attribution across published model schemas.
   - **Edge Node Inference (`/inference`):** Real-time per-flow classification with Top-3 feature attribution badges.

7. **Full-Stack Control Plane & Edge Console (`IBM Plex Sans` + Dark/Light UI):**
   - **HQ Model API (`:8000`):** FastAPI REST service for versioned ensemble distribution, metadata schemas, and federated update intake.
   - **HQ Web Console (`:8001`):** Light/Dark engineering UI with dataset-agnostic training, model dossiers, API endpoints panel, and 1-click **`Push to Edge`** deployment.
   - **Edge Client Console (`:5000`):** Protected by `ADMIN_KEY`, supports explicit HQ Push delivery, manual HQ pull, offline `.pt` upload, DP-SGD training, and local traffic inference.

8. **Dynamic Any-Dataset Compatibility:**
   - Adapts automatically to RT-IoT2022 (83 features), CIC-IDS2017 (78 features), NSL-KDD, UNSW-NB15, or custom CSV/Parquet schemas.

---

## Repository Structure

```
FedX-NIDS/
├── config.py                   # Central configuration (datasets, models, FL, DP, API)
├── main.py                     # 7-stage CLI training/FL/DP/XAI pipeline
├── requirements.txt            # Core Python dependencies
├── start_hq.py                 # Convenience launcher for HQ API + Dashboard
├── run_hq.sh / run_hq.bat      # Cross-platform HQ launchers
├── run_client.sh / .bat        # Cross-platform Edge Client launchers
│
├── models/                     # DL models, ensemble builders, evaluation
├── fl/                         # Federated learning strategies
├── dp/                         # DP-SGD clipping & noise accountant
├── xai/                        # Explainability modules
├── preprocessing/              # Dataset canonicalization & scaling
│
├── server_hq/                  # Central HQ (:8000 API & :8001 Dashboard)
│   ├── server_api.py           # Model serving & update upload API
│   ├── dashboard.py            # Web dashboard & 1-click Push to Edge
│   ├── central_aggregator.py   # Byzantine-robust federated aggregator
│   ├── registry.py             # SQLite version & client registry
│   └── templates/              # HQ UI templates (Light/Dark)
│
├── Client Server/              # Edge Client Node (:5000)
│   ├── client_app.py           # FastAPI edge server (DP training, inference, XAI)
│   └── templates/              # Edge UI templates (Light/Dark)
│
└── tests/                      # End-to-end verification tests
```

---

## Getting Started

### 1. Clone & Install Environment

```bash
git clone https://github.com/jehad005/FedX-NIDS.git
cd FedX-NIDS

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Running the Complete 7-Stage Pipeline

To run the full end-to-end pipeline (Preprocessing $\to$ 5-Fold CV $\to$ Top-2 Ensemble $\to$ 5 FL Methods $\to$ DP-SGD $\to$ SHAP XAI $\to$ HQ Model Publish):

```bash
python main.py --mode all --epochs 5 --kfold 5 --ensemble-method voting --voting-type soft
```

Command-line arguments:
- `--mode`: `all`, `demo`, `train`, `fl`, `dp`
- `--data`: Path to dataset CSV (if omitted, interactive selector launches)
- `--epochs`: Training epochs per candidate model
- `--kfold`: Folds for stratified cross-validation (default: `5`)
- `--ensemble-method`: `voting`, `stacking`, or `weighted`
- `--voting-type`: `soft` or `hard`

---

### 3. Launching the HQ Management Servers

Launch both the **API Server** and **Management Dashboard** with a single command:

```bash
# Linux / Kali Linux (convenience launcher)
./run_hq.sh

# Or direct Python launcher:
python start_hq.py
```

Or start them individually:

```bash
# Terminal 1: HQ API Server (Port 8000)
python -m server_hq.server_api

# Terminal 2: HQ Management Dashboard (Port 8001)
python -m server_hq.dashboard
```

- **HQ REST API:** [http://localhost:8000](http://localhost:8000)
- **HQ Web Dashboard:** [http://localhost:8001](http://localhost:8001)
- **Explainable AI Dashboard:** [http://localhost:8001/xai](http://localhost:8001/xai)
- **Model Updates & Aggregator:** [http://localhost:8001/updates](http://localhost:8001/updates)

---

### 4. Running an Edge Client Node

Start the Edge Client server cleanly via the cross-platform launcher:

```bash
# Linux / Kali Linux
./run_client.sh

# Windows CMD
run_client.bat
```

- **Edge Client Dashboard:** [http://localhost:5000](http://localhost:5000)

#### Edge Client Workflow:
1. **Model Deployment (HQ Push or Edge Pull):** Open HQ **Model Versions** (`http://localhost:8001/models`) and click **`Push to Edge`** on any published release to deploy directly to active Edge nodes (`Explicit Push Only`). Alternatively, on Edge `/sync`, click **`Sync Latest Model Now`** (or use **Advanced $\to$ Manual Model File Upload** for air-gapped `.pt` deployment).
2. **DP Local Training (`/training`):** Upload a local CSV traffic file. Train locally with DP-SGD (adjust Clipping Norm & Noise Multiplier). A protected parameter delta is created locally.
3. **Push Update (`/sync`):** Click **`Push DP Update to HQ`** to send clipped/noised parameter deltas $(\Delta W)$ to HQ without exposing raw traffic.
4. **Traffic Inference & On-Device XAI (`/inference`):** Upload traffic flow CSV logs to classify benign vs malicious traffic along with instant Top-3 threat indicator attributions.

#### Running Multiple Edge Clients Concurrently:
You can start multiple independent edge nodes on separate ports in seconds:

```bash
# Client Node 1 (Port 5000)
CLIENT_ID=client_node_01 ./run_client.sh 5000

# Client Node 2 (Port 5001)
CLIENT_ID=client_node_02 ./run_client.sh 5001
```

---

### 5. Federated Round Aggregation (HQ Side)

Once edge clients push their parameter updates, perform a federated round:

1. Navigate to **HQ Dashboard $\to$ Updates** (`http://localhost:8001/updates`).
2. Choose your desired aggregation algorithm:
   - `FedAvg`: Weighted sample averaging.
   - `Trimmed Mean`: Coordinate-wise trimming of extreme client updates (defends against gradient poisoning).
   - `Coordinate Median`: Median along each parameter dimension (defends against high-noise adversarial updates).
   - `Krum`: Geometric Euclidean distance selection (provably robust against up to $f$ Byzantine colluding clients).
3. Click **Aggregate Now**.
4. The system automatically creates and publishes the next global version (e.g. `v1.0.9`), preserving the active dataset's dynamic feature schema (`feature_names`), ready for edge clients to pull!

---

## 🔒 Security & Privacy Guarantees

- **Zero Raw Traffic Leakage:** Edge nodes never transmit packet payloads, IP headers, or raw CSV rows to the central server.
- **Admin Role Authorization (`ADMIN_KEY`):** Unauthorized users on edge nodes cannot deploy models, reconfigure HQ endpoints, or push federated updates (enforced with HTTP `403 Forbidden`).
- **Differential Privacy ($\varepsilon, \delta$):** Parameter updates are bounded by an $L_2$ clipping norm $C$ and perturbed with calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$.
- **Byzantine Fault Tolerance:** The central aggregator survives malicious/adversarial clients attempting model poisoning attacks.

---

## 📄 License
This project is open-source and intended for academic research and enterprise cybersecurity development.
