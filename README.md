# Distributed Pipeline Parallelism & Model Sharding

**Distributed Pipeline Parallelism** is a lightweight distributed training framework built with PyTorch that enables training models larger than a single GPU’s memory by sharding layers across multiple devices and executing them sequentially.

---

## Features

* 🚀 **Pipeline Parallelism**: Split models across multiple GPUs for scalable training
* 🧩 **Model Sharding**: Automatically partitions layers into pipeline stages
* ⚡ **Efficient GPU Utilization**: Each GPU handles only its assigned portion
* 🔄 **Forward & Backward Communication**: Handles activation and gradient passing between ranks
* 🌐 **Torch Distributed Integration**: Uses `torchrun`, NCCL, and Gloo backends
* 🧼 **Modular Design**: Clean abstraction for communication and sharding

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pipeline-parallelism.git

# Navigate to the project folder
cd pipeline-parallelism

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

* Python 3.9+
* PyTorch 2.0+
* NVIDIA GPU(s) with CUDA installed
* torchrun (included with PyTorch)

Verify installation:

```bash
python -c "import torch; print(torch.cuda.device_count())"
```

---

## Usage

Run the distributed pipeline using `torchrun`:

```bash
torchrun --nproc_per_node=NUM_GPUS train_pipeline.py
```

Example with 4 GPUs:

```bash
torchrun --nproc_per_node=4 train_pipeline.py
```

Each process will automatically:

* Initialize distributed environment
* Assign itself a GPU
* Load its shard of the model
* Communicate with neighboring ranks

---

## How It Works

### Model Sharding

The model is split into contiguous layer segments across GPUs.

Example with 8 layers and 4 GPUs:

| Rank   | Assigned Layers |
| ------ | --------------- |
| Rank 0 | Layers 0–1      |
| Rank 1 | Layers 2–3      |
| Rank 2 | Layers 4–5      |
| Rank 3 | Layers 6–7      |

---

### Forward Pass

Each rank:

1. Receives activations from previous rank
2. Computes forward pass on its shard
3. Sends activations to next rank

---

### Backward Pass

Each rank:

1. Receives gradients from next rank
2. Computes backward pass
3. Sends gradients to previous rank

---

## Repository Structure

```bash
pipeline-parallelism/
│
├── train_pipeline.py        # Main training script
├── pipeline_comms.py       # Communication abstraction
├── shard_model.py         # Model sharding logic
├── init_distributed.py    # Distributed setup
├── models/                # Model definitions
│
├── requirements.txt
└── README.md
```

---

## Distributed Initialization

Environment variables provided by `torchrun`:

* `RANK` — process ID
* `WORLD_SIZE` — total number of processes
* `LOCAL_RANK` — GPU index

Example:

```python
rank, world_size, device = init_distributed()
```

---

## Backend Support

* **NCCL**

  * Recommended for GPU training
  * High-performance communication

* **Gloo**

  * CPU fallback
  * Useful for debugging

---

## Example Workflow

```text
Input → Rank 0 → Rank 1 → Rank 2 → Rank 3 → Loss
         ↑                                  ↓
         ←────────── Gradients ────────────
```

---

## Why Pipeline Parallelism?

Benefits:

* Train larger models
* Reduce GPU memory usage
* Scale efficiently across multiple GPUs
* Foundation for training LLMs and large transformers

---

## Future Improvements

* Automatic model slicing from nn.Sequential
* Micro-batch pipeline support
* Async communication
* Checkpoint support
* Integration with HuggingFace models

---

## Contributing

Contributions are welcome.

You can improve:

* Performance optimization
* Model compatibility
* Pipeline scheduling

---


