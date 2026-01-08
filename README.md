# GAIA_agent

# GAIA AI Agent – Hugging Face Space

## Overview
This project implements a custom AI agent designed for the **GAIA benchmark**, deployed as a Hugging Face Space. The agent is built to handle real-world tasks that require structured reasoning, multi-step planning, tool usage, and precise answer formatting.

The repository mirrors the source code of a deployed Hugging Face Space. The authoritative execution environment is the hosted Space, where authentication, evaluation, and scoring are handled.

🔗 **Live Deployment:**  
https://huggingface.co/spaces/deni2004/deni_custom_agent

---

## What is GAIA?

GAIA is a benchmark designed to evaluate general-purpose AI assistants on real-world tasks that require a combination of core capabilities such as reasoning, multimodal understanding, web retrieval, and proficient tool use. It was introduced in the paper *“GAIA: A Benchmark for General AI Assistants”* and consists of **466 carefully curated questions**.

Although GAIA tasks are conceptually simple for humans, they remain remarkably challenging for current AI systems. Humans achieve an average success rate of approximately **92%**, while early large language models perform significantly lower. Recent agent-based systems show improved results, emphasizing the importance of planning, tool integration, and multi-step execution.

GAIA highlights current limitations of standalone LLMs and provides a rigorous benchmark for evaluating progress toward more general AI assistants.

---

## GAIA Core Principles

GAIA is designed around several key pillars:

- **Real-world difficulty:** Tasks require multi-step reasoning, multimodal understanding, and interaction with external tools.
- **Human interpretability:** Despite their difficulty for AI systems, questions remain easy for humans to follow and verify.
- **Non-gameability:** Correct answers require full task execution rather than pattern matching or brute-force approaches.
- **Simple evaluation:** Answers are concise, factual, and unambiguous, enabling reliable benchmarking.

---

## Difficulty Levels

GAIA tasks are organized into three levels of increasing complexity:

- **Level 1:** Fewer than five reasoning steps with minimal tool usage.
- **Level 2:** More complex reasoning requiring coordination between multiple tools and approximately 5–10 steps.
- **Level 3:** Long-horizon planning with advanced integration of multiple tools and modalities.

---

## How This Agent Works

- The agent is implemented in Python with a modular design
- A Gradio-based web interface is used for interaction
- Questions are fetched from a hosted GAIA backend
- The agent executes structured reasoning and tool-based logic
- Answers are submitted to the evaluation backend for scoring

Evaluation, authentication, and scoring are handled by the Hugging Face Spaces runtime.

---

## Deployment Model

This project is **designed to run inside the Hugging Face Spaces environment**.

The hosted Space provides:
- Hugging Face OAuth authentication
- Access to the GAIA evaluation backend
- Submission and scoring of agent outputs
- Leaderboard integration

Because of this, the **full evaluation and submission workflow does not execute in a purely local environment** without modification.

---

## Running Locally (Development Mode)

The source code can be run locally for development, inspection, and UI testing. Local execution is intended for understanding the agent architecture and iterating on logic, not for official evaluation or scoring.

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/deni_custom_agent.git
cd deni_custom_agent

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install gradio pandas requests
