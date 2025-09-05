import SETTINGS
import os

def update_settings_and_run(n):
    SETTINGS.NUM_FAULTS_TO_INJECT = n
    SETTINGS.FAULTS_TO_INJECT = -1

    print(f"\n[INFO] Running exhaustive FI with NUM_FAULTS_TO_INJECT = {n}")
    os.system("python main_online.py")  # oppure main_online.py
    os.system("python genGraph_online.py")  # oppure genGraph_online.py

for N in range(4, 6):
    update_settings_and_run(N)
