# PHD Project - Federated Learning System

A federated learning system built with Flower framework for distributed machine learning training across multiple clients.

## Project Overview

This project implements a federated learning system where:
- Multiple clients train models on their local data
- A central server coordinates the training process
- Models are aggregated using various strategies
- UV is used as the Python package manager

## Prerequisites

- Python 3.11 or higher
- UV package manager
- Git

## UV Installation

### Installing UV

UV is a fast Python package installer and resolver written in Rust. To install UV:

**On Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative installation methods:**
```bash
# Using pip (if you have Python installed)
pip install uv

# Using Homebrew (macOS/Linux)
brew install uv

# Using Cargo (if you have Rust installed)
cargo install uv
```

### Verifying UV Installation
```bash
uv --version
```

## Project Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd phd_project
   ```

2. **Install project dependencies with UV:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment (optional):**
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate    # Windows
   ```

## Running the System

### 1. Start the Tasks Worker
```bash
uv run ./tasks.py
```

### 2. Start the Monitor Server
```bash
uv run ./monitor_server.py
```

### 3. Start the Clients
```bash
./client.sh
```

## Project Structure

```
phd_project/
├── client.py              # Main client implementation
├── client.sh             # Client startup script
├── tasks.py              # Celery tasks worker
├── monitor_server.py     # Monitoring server
├── strategies/           # Federated learning strategies
├── data/                 # CSV data files for training
├── static/               # Static files for web interface
├── templates/            # HTML templates
├── pyproject.toml        # Project dependencies
└── uv.lock              # UV lock file
```

## Key Dependencies

- **flwr[simulation]**: Flower framework for federated learning
- **tensorflow**: Machine learning framework
- **torch**: PyTorch framework
- **celery**: Distributed task queue
- **fastapi**: Web framework for monitoring server
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning utilities

## Usage Examples

### Running Individual Components

**Start a single client:**
```bash
uv run client.py --server-url "127.0.0.1:8789" --data ./data/csv1.csv
```

**Run with verbose logging:**
```bash
uv run client.py --server-url "127.0.0.1:8789" --data ./data/csv2.csv --verbose
```

### Development Commands

**Add a new dependency:**
```bash
uv add package-name
```

**Update dependencies:**
```bash
uv sync
```

**Run tests:**
```bash
uv run pytest
```

## Troubleshooting

1. **UV not found:** Make sure UV is installed and in your PATH
2. **Port conflicts:** Change ports in the configuration if default ports are in use
3. **Dependency issues:** Run `uv sync` to ensure all dependencies are properly installed

## Contributing

1. Install development dependencies with UV
2. Follow the coding standards
3. Test changes thoroughly
4. Update documentation as needed

## License

This project is part of a PhD research project.