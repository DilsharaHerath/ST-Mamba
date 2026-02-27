#!/bin/bash

# Script to start Mamba-Irradiance training in a tmux session
# Usage: bash start_training.sh

# Set working directory
cd /storage2/CV_Irradiance/scripts/Mamba-Irradiance/SolarMamba

# Create directories if they don't exist
mkdir -p logs checkpoints

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "Starting training session in tmux 'oshadha'..."
echo "Logs will be saved to: $LOG_FILE"
echo "Checkpoints will be saved to: checkpoints/"
echo "Best model will be saved as: checkpoints/best_model.pth"
echo ""
echo "To attach to the session: tmux attach -t oshadha"
echo "To detach from the session: Press Ctrl+B then D"
echo "To view logs: tail -f $LOG_FILE"
echo ""

# Install mambavision package if not already installed
cd /storage2/CV_Irradiance/scripts/Mamba-Irradiance
pip install -e . > /dev/null 2>&1

# Return to SolarMamba directory
cd /storage2/CV_Irradiance/scripts/Mamba-Irradiance/SolarMamba

# Start training in tmux session
tmux new-session -d -s oshadha bash -c "\
    source ~/.bashrc; \
    conda activate solar_mamba_env; \
    cd /storage2/CV_Irradiance/scripts/Mamba-Irradiance/SolarMamba; \
    echo 'Training started at $(date)' | tee $LOG_FILE; \
    echo 'Using conda environment: solar_mamba_env' | tee -a $LOG_FILE; \
    echo 'Using GPU: 0' | tee -a $LOG_FILE; \
    echo '===========================================' | tee -a $LOG_FILE; \
    CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml --output_dir checkpoints 2>&1 | tee -a $LOG_FILE; \
    echo '===========================================' | tee -a $LOG_FILE; \
    echo 'Training completed/stopped at $(date)' | tee -a $LOG_FILE; \
    exec bash"

echo "Training started successfully!"
echo "Session name: oshadha"
tmux ls | grep oshadha
