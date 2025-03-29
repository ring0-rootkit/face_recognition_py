.PHONY: install train clean run help

# Default target
all: install

# Show help
help:
	@echo "Available targets:"
	@echo "  install    - Install project dependencies"
	@echo "  train      - Train the model on LFW dataset"
	@echo "  clean      - Remove temporary files and caches"
	@echo "  run        - Run the GUI application"
	@echo "  help       - Show this help message"

# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# Train the model
train:
	@if [ ! -f "datasets/lfw.bin" ]; then \
		echo "Error: LFW dataset not found at datasets/lfw.bin"; \
		echo "Please download the dataset first"; \
		exit 1; \
	fi
	python train_ghostface.py --lfw_path datasets/lfw.bin --epochs 100 --batch_size 32

# Run the GUI application
run:
	python gui.py

# Create necessary directories
setup:
	mkdir -p datasets
	mkdir -p faces
	mkdir -p checkpoints

# Download LFW dataset (placeholder - you need to implement actual download)
download_dataset:
	@echo "Please download the LFW dataset manually and place it in datasets/lfw.bin"
	@echo "You can download it from: http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
	@echo "After downloading, extract and convert to binary format"

# Full setup including dataset
setup_full: setup download_dataset install

# Train with default parameters
train_default: train

# Train with custom parameters
train_custom:
	@read -p "Enter number of epochs (default: 100): " epochs; \
	read -p "Enter batch size (default: 32): " batch_size; \
	python train_ghostface.py --lfw_path datasets/lfw.bin --epochs $${epochs:-100} --batch_size $${batch_size:-32}

# Run with GPU support
run_gpu:
	CUDA_VISIBLE_DEVICES=0 python gui.py

# Run with CPU only
run_cpu:
	CUDA_VISIBLE_DEVICES=-1 python gui.py 