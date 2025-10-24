# Abstract Agent Runner

<br>

## Introduction

Abstract Agent Runner is a comprehensive testing framework designed to evaluate AI coding agents against standardized programming challenges. The system provides a sandboxed environment where agents can attempt to solve problems from well-established benchmark suites, including Polyglot programming challenges and SWE-Bench Verified real-world software engineering tasks.

The framework isolates agent execution in Docker containers, ensuring safe and reproducible testing while providing agents with access to complete development environments. It supports multiple problem domains, from algorithmic puzzles to complex software maintenance tasks, making it an ideal tool for researchers and developers working on automated code generation and program synthesis.

<br>



## Requirements

- **Docker**: Required for creating isolated sandbox environments
- **Python 3.11+**: For running the testbed framework
- **UV**: Python package manager for virtual environment management
- **Git**: Required for repository operations and dataset management
- **Chutes**: Needed for inference and embedding

<br>



## Setup

### Creating Virtual Environment with UV

First, create and activate a virtual environment using UV:

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies in the virtual environment
uv pip install -e .
```

### Setup local inference gateway

To learn more about the inference gateway terminology and its purpose, see `docs/proxy_and_gateway.md`.

```bash
cd inference_gateway
cp .env.example .env
```

Add your Chutes API key to the `CHUTES_API_KEY` line in the `.env` file.

In a new terminal, run

```bash
python main.py
```

Lastly, get your local IP address by running `ifconfig en0` (or another network interface if not on macOS), and looking at the line after `inet`. This will be used for the `<gateway_url>` argument. Do not use your public IP (which you can find online), or the loopback address (`127.0.0.1`), or the all-interfaces address (`0.0.0.0`). It must be the local IP address, which is typically similar to `192.168.x.x` or `10.x.x.x` (you can look up private IP ranges to know what to look for).

## Usage

The framework is controlled through the CLI located in the root directory. The basic usage pattern is:

```bash
python cli.py <suite_name> <problem_name> <agent_file> <gateway_url>
```

### Examples

**Running an agent on a Polyglot problem:**
```bash
python cli.py polyglot affine-cipher test_agent.py http://<IP_ADDRESS>:8000
```

**Running an agent on a SWE-Bench problem:**
```bash
python cli.py swebench_verified django__django-12308 test_agent.py http://<IP_ADDRESS>:8000
```

**Available suites:**
- `polyglot`: Algorithmic programming challenges
- `swebench_verified`: Real-world software engineering tasks

**Optional flags:**
- `--log-docker-to-stdout`: Print Docker container logs to stdout in real-time
- `--include-solution`: Expose the solution to the agent at `/sandbox/solution.diff`
- `--verbose`: Enable verbose (debug) logging

**Examples with flags:**
```bash
python cli.py polyglot affine-cipher test_agent.py --include-solution --log-docker-to-stdout --verbose
```

### Agent Implementation

See `docs/agent_environment.md` for a description of how agents should look like and what their environment exposes.