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

### 1. Prerequisites
- **Python:** 3.9, 3.10, 3.11, or 3.12 (Make sure Python is added to system PATH during installation).
- **Git:** Installed on your system.

---

### 2. Clone & Install Environment

Choose your operating system below:

#### 🪟 Windows (Command Prompt or PowerShell)

**Option A: 1-Click Automated Setup (Recommended)**
Double-click `setup_and_run.bat` or run:
```cmd
setup_and_run.bat setup
```

**Option B: Manual Setup via Command Prompt (CMD)**
```cmd
git clone https://github.com/jehad005/FedX-NIDS.git
cd FedX-NIDS

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Option C: Manual Setup via PowerShell**
```powershell
git clone https://github.com/jehad005/FedX-NIDS.git
cd FedX-NIDS

# Create virtual environment
python -m venv venv

# If PowerShell script execution is restricted, allow it once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> **PyTorch on Windows:** `pip install -r requirements.txt` installs standard PyTorch wheels. If you have an NVIDIA GPU and want CUDA acceleration, or want a lightweight CPU-only build, you can optionally run:
> ```cmd
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

#### 🐧 Linux / macOS / Kali Linux

**Option A: Automated Setup**
```bash
chmod +x setup_and_run.sh run_hq.sh run_client.sh
./setup_and_run.sh setup
```

**Option B: Manual Setup**
```bash
git clone https://github.com/jehad005/FedX-NIDS.git
cd FedX-NIDS

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Running the Complete 7-Stage Pipeline

To run the full end-to-end pipeline (Preprocessing $\to$ 5-Fold CV $\to$ Top-2 Ensemble $\to$ 5 FL Methods $\to$ DP-SGD $\to$ SHAP XAI $\to$ HQ Model Publish):

```bash
# Windows / Linux:
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

### 4. Launching the HQ Management Servers

Launch both the **API Server (Port 8000)** and **Management Dashboard (Port 8001)**:

#### 🪟 Windows:
```cmd
# Double click run_hq.bat or run in CMD:
run_hq.bat

# Or direct Python launcher:
python start_hq.py
```

#### 🐧 Linux / macOS:
```bash
./run_hq.sh

# Or direct Python launcher:
python start_hq.py
```

#### Starting Individually (Any OS):
```bash
# Terminal 1: HQ API Server (Port 8000)
python -m server_hq.server_api

# Terminal 2: HQ Management Dashboard (Port 8001)
python -m server_hq.dashboard
```

- **HQ REST API:** [http://localhost:8000](http://localhost:8000) (Swagger Docs: `/docs`)
- **HQ Web Dashboard:** [http://localhost:8001](http://localhost:8001)
  - **Admin Credentials:** Username: `admin` | Password: `admin123`
- **Federated Models:** [http://localhost:8001/fl-models](http://localhost:8001/fl-models)
- **Deep Learning Baseline Models:** [http://localhost:8001/models](http://localhost:8001/models)
- **Explainable AI Dashboard:** [http://localhost:8001/xai](http://localhost:8001/xai)
- **Model Updates & Aggregator:** [http://localhost:8001/updates](http://localhost:8001/updates)

---

### 5. Running an Edge Client Node

Start the Edge Client server on `http://localhost:5000`:

#### 🪟 Windows:
```cmd
# Standard launch (Port 5000):
run_client.bat

# Run with custom port:
run_client.bat 5001

# Run with custom Client ID (CMD):
set CLIENT_ID=client_node_01 && run_client.bat 5000

# Run with custom Client ID (PowerShell):
$env:CLIENT_ID="client_node_01"; .\run_client.bat 5000
```

#### 🐧 Linux / macOS:
```bash
# Standard launch (Port 5000):
./run_client.sh

# Run with custom port:
./run_client.sh 5001

# Run with custom Client ID:
CLIENT_ID=client_node_01 ./run_client.sh 5000
```

- **Edge Client Dashboard:** [http://localhost:5000](http://localhost:5000)

#### Edge Client Workflow:
1. **Model Deployment (1-Click HQ Push or Edge Pull):** Open HQ **Models** (`/models` or `/fl-models`) and click **`Push to Client Edge`** on any published release to deploy directly to active Edge nodes (`:5000` or registered nodes). Alternatively, on Edge `/sync`, click **`Sync Latest Model Now`**.
2. **DP Local Training (`/training`):** Upload a local CSV traffic file. Train locally with DP-SGD (adjust Clipping Norm & Noise Multiplier). A protected parameter delta is created locally.
3. **Push Update (`/sync`):** Click **`Push DP Update to HQ`** to send clipped/noised parameter deltas $(\Delta W)$ to HQ without exposing raw traffic.
4. **Traffic Inference & On-Device XAI (`/inference`):** Upload traffic flow CSV logs to classify benign vs malicious traffic along with instant Top-3 threat indicator attributions.

---

### 6. Federated Round Aggregation & Byzantine Defense (HQ Side)

Once edge clients push their parameter updates, perform a federated round:

1. Navigate to **HQ Dashboard $\to$ Updates** (`http://localhost:8001/updates`).
2. Choose your desired aggregation algorithm:
   - `★ All Models (Evaluate All 9 & Select Best)`: Evaluates all 9 algorithms on holdout data and crowns the top-performing model as winner.
   - `FedAvg`: Weighted sample averaging.
   - `FedProx`: Proximal regularizer for Non-IID Dirichlet drift and DP noise.
   - `SCAFFOLD`: Variance reduction using control variates.
   - `FedNova`: Normalized averaging handling heterogeneous local steps.
   - `MOON`: Model-contrastive federated representation learning.
   - `Trimmed Mean`: Coordinate-wise trimming of extreme client updates (defends against gradient poisoning).
   - `Coordinate Median`: Median along each parameter dimension (defends against high-noise adversarial updates).
   - `Krum` / `Multi-Krum`: Geometric Euclidean distance selection (provably robust against colluding Byzantine clients).
3. Click **Aggregate Now**.
4. The system automatically evaluates candidates, logs leaderboard statistics, and publishes the next global version (`FL Global Model`, e.g. `v2.0.3`), preserving dynamic feature schemas ready for edge nodes to pull or receive!

---

## 🔒 Security & Privacy Guarantees

- **Zero Raw Traffic Leakage:** Edge nodes never transmit packet payloads, IP headers, or raw CSV rows to the central server.
- **Admin Role Authorization (`ADMIN_KEY`):** Unauthorized users on edge nodes cannot deploy models, reconfigure HQ endpoints, or push federated updates (enforced with HTTP `403 Forbidden`).
- **Differential Privacy ($\varepsilon, \delta$):** Parameter updates are bounded by an $L_2$ clipping norm $C$ and perturbed with calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$.
- **Byzantine Fault Tolerance:** The central aggregator survives malicious/adversarial clients attempting model poisoning attacks.

---

## 📄 License
This project is open-source and intended for academic research and enterprise cybersecurity development.
