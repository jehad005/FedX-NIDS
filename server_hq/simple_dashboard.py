#!/usr/bin/env python3
"""
Simple SecureFed-NIDS Dashboard - Standalone version without fastapi dependency
"""

import os
import sys
import json
import sqlite3
import uuid
import mimetypes
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import cgi

ROOT = Path(__file__).parent.parent
PORT = 8001

class SimpleDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging

    def send_html(self, content, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(content.encode())

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/':
            self.send_html(self.get_index())
        elif path == '/training':
            self.send_html(self.get_training_page())
        elif path.startswith('/api/dashboard/training/jobs'):
            self.send_json(self.get_jobs())
        else:
            self.send_html(self.get_index())

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith('/api/dashboard/training/upload'):
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' in content_type:
                job_id = str(uuid.uuid4())
                save_data = {
                    'status': 'success',
                    'job_id': job_id,
                    'message': f'Dataset uploaded; training started (simulated). Job ID: {job_id}'
                }
                self.send_json(save_data)
            else:
                self.send_json({'status': 'error', 'message': 'Expected multipart/form-data'}, 400)
        elif path.startswith('/api/dashboard/training/status/'):
            job_id = path.split('/')[-1]
            self.send_json({'status': 'success', 'job': {'job_id': job_id, 'status': 'completed', 'progress': 100}})
        elif path.startswith('/api/dashboard/training/stop/'):
            job_id = path.split('/')[-1]
            self.send_json({'status': 'success', 'job_id': job_id, 'message': 'Stop requested'})
        elif path.startswith('/api/dashboard/training/cleanup/'):
            job_id = path.split('/')[-1]
            self.send_json({'status': 'success', 'job_id': job_id, 'deleted': ['log', 'dataset']})
        else:
            self.send_json({'status': 'error', 'message': 'Not found'}, 404)

    def get_jobs(self):
        db_path = ROOT / 'data' / 'securefed_nids.db'
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT job_id, dataset_name, status, progress, message FROM training_jobs ORDER BY created_at DESC LIMIT 10')
                rows = cursor.fetchall()
                jobs = [{'job_id': r[0], 'dataset_name': r[1], 'status': r[2], 'progress': r[3], 'message': r[4]} for r in rows]
                return {'status': 'success', 'jobs': jobs}
            except:
                pass
            conn.close()
        return {'status': 'success', 'jobs': []}

    def get_index(self):
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SecureFed-NIDS HQ Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4">
    <h1 class="mb-4">🔒 SecureFed-NIDS HQ Dashboard</h1>
    <p class="text-muted">Distributed Intrusion Detection System</p>

    <div class="row mb-4">
        <div class="col">
            <a href="/training" class="btn btn-primary btn-lg">📊 Training Dashboard</a>
        </div>
        <div class="col-auto">
            <button onclick="location.reload()" class="btn btn-outline-secondary btn-lg">🔄 Refresh</button>
        </div>
    </div>

    <div class="card">
        <div class="card-header">📋 Recent Training Jobs</div>
        <div class="card-body" id="jobs">
            <p class="text-muted">Loading job data...</p>
        </div>
    </div>
</div>
<script>
fetch('/api/dashboard/training/jobs')
    .then(r => r.json())
    .then(data => {
        const jobsDiv = document.getElementById('jobs');
        if (data.jobs.length === 0) {
            jobsDiv.innerHTML = '<p class="text-muted">No training jobs yet.</p>';
        } else {
            jobsDiv.innerHTML = data.jobs.map(j =>
                `<div class="border p-2 mb-2">
                    <strong>${j.job_id.slice(0,8)}...</strong> - ${j.dataset_name || 'Unknown'}
                    <br><span class="badge bg-${j.status === 'completed' ? 'success' : j.status === 'running' ? 'warning' : 'secondary'}">${j.status}</span>
                    <progress class="progress ms-2" value="${j.progress}" max="100" style="width: 100px;"></progress>
                </div>'
            ).join('');
        }
    })
    .catch(e => console.log('Jobs data unavailable'));
</script>
</body>
</html>'''

    def get_training_page(self):
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training - SecureFed-NIDS</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2>🧠 Model Training</h2>
        <a href="/" class="btn btn-outline-secondary">← Dashboard</a>
    </div>

    <div class="card mb-4">
        <div class="card-header">📁 Upload Dataset</div>
        <div class="card-body">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="mb-3">
                    <label class="form-label">Dataset Name</label>
                    <input type="text" name="dataset_name" class="form-control" required placeholder="e.g., NIDS Training Data">
                </div>
                <div class="mb-3">
                    <label class="form-label">Format</label>
                    <select name="dataset_format" class="form-select">
                        <option value="csv">CSV</option>
                        <option value="parquet">Parquet</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label class="form-label">Target Column</label>
                    <input type="text" name="target_column" class="form-control" placeholder="label">
                </div>
                <div class="mb-3">
                    <label class="form-label">Epochs</label>
                    <input type="number" name="epochs" class="form-control" value="3" min="1">
                </div>
                <div class="mb-3">
                    <label class="form-label">K-Fold</label>
                    <input type="number" name="kfold" class="form-control" value="5" min="1" max="10">
                </div>
                <div class="mb-3">
                    <label class="form-label">Dataset File (CSV/Parquet)</label>
                    <input type="file" name="file" class="form-control" accept=".csv,.parquet" required>
                </div>
                <button type="submit" class="btn btn-primary">🚀 Start Training</button>
            </form>
        </div>
    </div>

    <div class="card">
        <div class="card-header">📊 Training Status</div>
        <div class="card-body">
            <p class="text-muted">Upload a dataset above to start training.</p>
        </div>
    </div>
</div>
<script>
document.getElementById('uploadForm').onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const resp = await fetch('/api/dashboard/training/upload', {
        method: 'POST',
        body: formData
    });
    const data = await resp.json();
    alert('Job started: ' + (data.job_id || 'check console'));
};
</script>
</body>
</html>'''

def run_server(port=PORT):
    url = f'http://localhost:{port}'
    print(f'\n{"="*60}')
    print('SECUREFED-NIDS SIMPLE DASHBOARD')
    print(f'{"="*60}')
    print(f'Dashboard URL: http://localhost:{port}')
    print(f'API Server: http://localhost:8000 (if separately running)')
    print(f'{"="*60}\n')
    print('Press Ctrl+C to stop')
    print(f'{"="*60}\n')

    server = HTTPServer(('0.0.0.0', port), SimpleDashboardHandler)
    server.serve_forever()

if __name__ == '__main__':
    run_server()