# Getting Started

## Installation

### Setup with uv

```bash
# Clone Repository
git clone https://github.com/Imageomics/hpc-inference.git
cd hpc-inference

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install package
uv venv hpc-inference-env
source hpc-inference-env/bin/activate

# Install base package
uv pip install -e .
```

### Verify Installation

```bash
# Test the installation
python -c "import hpc_inference; print('✓ HPC-Inference installed successfully')"
```

### Optional Dependencies

The package comes with ready-to-use job scripts for various use cases. Install additional dependencies based on your needs:

```bash
# Check installation status and available features
python -c "from hpc_inference import print_installation_guide; print_installation_guide()"

# Install specific feature sets
uv pip install -e ".[openclip]"     # For CLIP embedding
uv pip install -e ".[detection]"    # For face/animal detection  
uv pip install -e ".[all]"          # Install dependency for all use cases
```

## Quick Start

### Basic Usage



### Next Steps

- Explore the [API Reference](api-reference.md) for detailed documentation
