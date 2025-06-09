import os
import subprocess

# Configuration
stage = -1
stop_stage = 3

data_url = "www.openslr.org/resources/60"
data_dir = "/mnt/lyuxiang.lx/data/tts/openslr/libritts"
pretrained_model_dir = "../../../pretrained_models/CosyVoice-300M"

parts = ["dev-clean", "test-clean", "dev-other", "test-other", "train-clean-100", "train-clean-360", "train-other-500"]

def run_command(command):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

# Stage -1: Data Download
if stage <= -1 and stop_stage >= -1:
    print("Stage -1: Data Download")
    for part in parts:
        run_command(["bash", "local/download_and_untar.sh", data_dir, data_url, part])