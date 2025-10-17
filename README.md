# Tuilink Project - AI Auto Reply

An intelligent system for generating contextually appropriate replies in professional conversations, particularly focused on job referral scenarios.

## Overview

This project implements an automated reply generation system that:

1. Analyzes conversation context
2. Classifies conversation categories
3. Suggests relevant topics for replies
4. Generates professional and contextually appropriate responses

## Project Structure

```
.
├── charts/                     # Mermaid charts
├── input/                      # Input data directory
│   ├── categories.json         # Category definitions
│   └── convo_2454_rows.xlsx    # Conversation dataset
├── models/                     # Core data models
├── nodes/                      # Processing nodes
├── output/                     # Generated outputs
├── utils/                      # Utility functions
└── *.ipynb                     # Local notebooks to play around
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The project uses environment variables for configuration. Create a `.env` file with the following necessary credentials and settings:

```bash
# OpenAI
OPENAI_API_KEY=<your-openai-api-key>

# LLM configuration
LLM_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0

# Caching, whether to cache the LLM responses locally
LLM_USE_CACHE=false

# Whether to show debug fields in outputs (reason, confidence)
LLM_INCLUDE_DEBUG_FIELDS=true
```

## Build for Deployment

Use the build script to create a self-contained `dist/` folder ready for packaging and deployment.

1. Ensure your `.env` is present and set `LLM_USE_CACHE=false` and `LLM_INCLUDE_DEBUG_FIELDS=false` for deployment.

2. Run the build script:

```bash
chmod +x ./build.sh
./build.sh
```

3. The script will create `./dist` containing:

- `handler.py`
- `requirements.txt` (copied from `requirements-compact.txt`)
- `models/`, `nodes/`, `utils/`, `input/` (project files only)
- `__init__.py` files for Python package directories (`models/`, `nodes/`, `utils/`)

4. Deployment notes for your infra repo:

- Set `AI_AUTO_REPLY_LAMBDA_SOURCE_PATH` to be the path to the `dist` folder of this repo, e.g. `../../course-ai-auto-reply-starter/dist/`

- Set `AI_AUTO_REPLY_LAMBDA_HANDLER` to be `handler.handler`
