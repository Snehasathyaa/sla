#!/usr/bin/env python3
"""
One-click startup: generates dataset, trains models, then launches Flask
"""
import os
import sys
import subprocess

print("=" * 60)
print("  SMART SLA PREDICTION SYSTEM — KERALA GOVERNMENT")
print("=" * 60)

# 1. Install dependencies
print("\n[1/4] Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt',
                '-q', '--break-system-packages'], check=False)

# 2. Generate dataset
if not os.path.exists('dataset/grievance_dataset.csv'):
    print("\n[2/4] Generating dataset...")
    subprocess.run([sys.executable, 'generate_dataset.py'], check=True)
else:
    print("\n[2/4] Dataset already exists, skipping generation...")

# 3. Train models
if not os.path.exists('models/sla_classifier.pkl'):
    print("\n[3/4] Training ML models (this may take 1-2 minutes)...")
    subprocess.run([sys.executable, 'train_model.py'], check=True)
else:
    print("\n[3/4] Models already trained, skipping...")

# 4. Launch Flask
print("\n[4/4] Starting Flask application...")
print("\n  ➜ Open http://127.0.0.1:5000")
print("  ➜ Admin: admin@kerala.gov.in / admin123")
print("  ➜ Official: officer1@kerala.gov.in / officer123")
print("  ➜ Citizen: citizen1@example.com / citizen123\n")

os.system(f'{sys.executable} app.py')
