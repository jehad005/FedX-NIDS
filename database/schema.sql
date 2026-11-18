CREATE TABLE client_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    base_version TEXT NOT NULL,
                    round INTEGER DEFAULT 0,
                    samples INTEGER DEFAULT 0,
                    weights_path TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'received', aggregated_version TEXT, aggregation_method TEXT, aggregated_at TEXT, archive_path TEXT, dp_clipping_norm REAL, dp_noise_multiplier REAL, local_steps INTEGER DEFAULT 1,
                    FOREIGN KEY(client_id) REFERENCES clients(client_id)
                );
CREATE TABLE clients (
                    client_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    registered_at TEXT NOT NULL,
                    last_seen TEXT,
                    last_version TEXT
                );
CREATE TABLE dashboard_clients (
                client_id TEXT PRIMARY KEY,
                client_name TEXT,
                ip_address TEXT,
                last_seen TIMESTAMP,
                status TEXT DEFAULT 'active',
                model_version TEXT,
                download_count INTEGER DEFAULT 0,
                upload_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
CREATE TABLE dashboard_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT,
                version_from TEXT,
                version_to TEXT,
                update_size INTEGER,
                dp_epsilon REAL,
                accuracy REAL,
                status TEXT DEFAULT 'pending',
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES dashboard_clients (client_id)
            );
CREATE TABLE evaluation_plots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    plot_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL
                );
CREATE TABLE fl_leaderboard (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_idx INTEGER NOT NULL,
                    base_version TEXT NOT NULL,
                    resulting_version TEXT,
                    method TEXT NOT NULL,
                    accuracy REAL,
                    f1 REAL,
                    auc REAL,
                    is_winner INTEGER DEFAULT 0,
                    recorded_at TEXT NOT NULL
                );
CREATE TABLE fl_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
CREATE TABLE model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL DEFAULT 'SecureFed-Global-Ensemble',
                    architecture TEXT,
                    fl_method TEXT,
                    input_dim INTEGER,
                    num_classes INTEGER,
                    accuracy REAL,
                    weights_path TEXT NOT NULL,
                    package_path TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    is_latest INTEGER NOT NULL DEFAULT 0
                );
CREATE TABLE training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    val_loss REAL,
                    train_acc REAL,
                    val_acc REAL,
                    learning_rate REAL,
                    recorded_at TEXT NOT NULL
                );
CREATE TABLE training_jobs (
                job_id TEXT PRIMARY KEY,
                dataset_name TEXT,
                dataset_path TEXT,
                dataset_format TEXT,
                target_column TEXT,
                epochs INTEGER DEFAULT 3,
                kfold INTEGER DEFAULT 5,
                ensemble_method TEXT DEFAULT 'voting',
                voting_type TEXT DEFAULT 'soft',
                status TEXT DEFAULT 'queued',
                progress INTEGER DEFAULT 0,
                message TEXT,
                log_path TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_version TEXT,
                plot_paths TEXT
            , model_metrics TEXT, kfold_results TEXT, statistical_results TEXT, ensemble_metrics TEXT);
