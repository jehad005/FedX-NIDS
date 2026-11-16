# SecureFed-NIDS Edge Client Server

Isolated client server node for federated learning, differential privacy, and intrusion detection inference.

## Features
- **Central HQ Sync**: Pull the versioned ensemble model (`FL Global Model`) from HQ (`http://localhost:8000`) via compressed zlib+base64 transfer.
- **DP-SGD Local Training**: Train locally on private network traffic CSVs with per-sample gradient clipping ($C$) and Gaussian noise multiplier ($\sigma$).
- **Secure Weight Updates**: Transmit differentially private weight updates back to HQ without leaking raw packets, logs, or sensitive data.
- **Inference & Explainability**: Upload CSV traffic records to classify attacks vs normal flows with prediction confidence scores.

---

## Running the Client

### Option A: From Repository Root (Recommended)

- **Windows (CMD / PowerShell):**
  ```cmd
  run_client.bat
  ```
  *(To specify port or client ID: `set CLIENT_ID=edge_01 && run_client.bat 5000`)*

- **Linux / macOS:**
  ```bash
  ./run_client.sh
  ```
  *(To specify port or client ID: `CLIENT_ID=edge_01 ./run_client.sh 5000`)*

---

### Option B: From Inside "Client Server" Directory

#### Windows (Command Prompt)
```cmd
cd "Client Server"
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
python client_app.py
```

#### Windows (PowerShell)
```powershell
cd "Client Server"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python client_app.py
```

#### Linux / macOS (Bash)
```bash
cd "Client Server"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python client_app.py
```

Open your browser at `http://localhost:5000`.
