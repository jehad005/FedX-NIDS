# SecureFed-NIDS Edge Client Server

Isolated client server node for federated learning, differential privacy, and intrusion detection inference.

## Features
- **Central HQ Sync**: Pull the versioned ensemble model (`EnsembleTop2`) from HQ (`http://localhost:8000`) via compressed zlib+base64 transfer.
- **DP-SGD Local Training**: Train locally on private network traffic CSVs with per-sample gradient clipping ($C$) and Gaussian noise multiplier ($\sigma$).
- **Secure Weight Updates**: Transmit differentially private weight updates back to HQ without leaking raw packets, logs, or sensitive data.
- **Inference & Explainability**: Upload CSV traffic records to classify attacks vs normal flows with prediction confidence scores.

## Running the Client
```bash
cd "Client Server"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python client_app.py
```
Open your browser at `http://localhost:5000`.