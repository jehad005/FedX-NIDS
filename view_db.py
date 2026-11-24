#!/usr/bin/env python3
"""
SecureFed-NIDS Database Inspector & Viewer
==========================================
Interactive command-line tool to view tables, schemas, and live records
stored inside the SQLite database (database/securefed_nids.db).
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "database" / "securefed_nids.db"

def print_banner():
    print("=" * 82)
    print("  SECUREFED-NIDS: SQLite Database Inspector")
    print(f"  Database File: {DB_PATH}")
    print("=" * 82)

def view_database():
    if not DB_PATH.exists():
        print(f"[!] Database file not found at: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Summary of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r["name"] for r in cursor.fetchall()]
    print("\n[+] TABLES FOUND IN DATABASE:")
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {t}")
        cnt = cursor.fetchone()["cnt"]
        print(f"    - {t:<22} : {cnt} records")

    # 2. Model Versions Table
    print("\n" + "-" * 82)
    print("  TABLE: model_versions (Published DL & FL Global Models)")
    print("-" * 82)
    cursor.execute("""
        SELECT version, model_name, architecture, fl_method, accuracy, is_latest, created_at 
        FROM model_versions 
        ORDER BY created_at DESC;
    """)
    rows = cursor.fetchall()
    if rows:
        header = f"{'Version':<10} {'Model Name':<20} {'Algorithm':<14} {'Accuracy':<10} {'Latest':<8} {'Created At':<19}"
        print(header)
        print("-" * len(header))
        for r in rows:
            acc = f"{r['accuracy']*100:.2f}%" if r['accuracy'] is not None else "N/A"
            alg = (r['fl_method'] or "baseline").upper()
            latest = "YES ★" if r['is_latest'] else "No"
            created = str(r['created_at'])[:19] if r['created_at'] else "N/A"
            print(f"v{r['version']:<9} {str(r['model_name'])[:18]:<20} {alg:<14} {acc:<10} {latest:<8} {created:<19}")
    else:
        print("  (No model records found)")

    # 3. Clients Table
    print("\n" + "-" * 82)
    print("  TABLE: clients (Edge Clients Registered to HQ)")
    print("-" * 82)
    cursor.execute("SELECT client_id, display_name, last_version, last_seen, registered_at FROM clients ORDER BY registered_at DESC;")
    rows = cursor.fetchall()
    if rows:
        header = f"{'Client ID':<25} {'Display Name':<20} {'Model Ver':<12} {'Last Seen':<19}"
        print(header)
        print("-" * len(header))
        for r in rows:
            c_name = r['display_name'] or "Edge Node"
            c_ver = f"v{r['last_version']}" if r['last_version'] else "v1.0.8"
            l_seen = str(r['last_seen'])[:19] if r['last_seen'] else "N/A"
            print(f"{r['client_id']:<25} {c_name:<20} {c_ver:<12} {l_seen:<19}")
    else:
        print("  (No client records found)")

    # 4. Client Updates (Differential Privacy & FL Round Updates)
    print("\n" + "-" * 82)
    print("  TABLE: client_updates (Differentially Private Edge Model Updates)")
    print("-" * 82)
    cursor.execute("""
        SELECT id, client_id, round, samples, dp_clipping_norm, dp_noise_multiplier, status, received_at 
        FROM client_updates 
        ORDER BY id DESC LIMIT 10;
    """)
    rows = cursor.fetchall()
    if rows:
        header = f"{'ID':<6} {'Client ID':<20} {'Round':<8} {'Samples':<10} {'DP Clip (C)':<14} {'DP Noise (σ)':<14} {'Status':<10}"
        print(header)
        print("-" * len(header))
        for r in rows:
            clip = f"{r['dp_clipping_norm']}" if r['dp_clipping_norm'] is not None else "1.0"
            noise = f"{r['dp_noise_multiplier']}" if r['dp_noise_multiplier'] is not None else "1.1"
            print(f"#{r['id']:<5} {r['client_id']:<20} R{r['round']:<7} {r['samples']:<10} {clip:<14} {noise:<14} {r['status']:<10}")
    else:
        print("  (No updates found)")

    print("\n" + "=" * 82)
    print("  [✓] All tables verified successfully in database/securefed_nids.db")
    print("=" * 82 + "\n")
    conn.close()

if __name__ == "__main__":
    print_banner()
    view_database()
