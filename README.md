# 🤖 Agentic Data Analyst

An AI-powered data analysis assistant that turns natural language questions into executable Python code. Chat with your CSV data, get instant insights, and export results—all with full transparency into the agent's reasoning and generated code.

## How to run it

### Option 1: Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
echo "OPENAI_MODEL=gpt-5" >> .env  # Defaults to gpt-5

# Run the agent CLI (uses default CSVs in current directory)
python -m agentic_analyst.cli_agent

# Or specify custom CSVs and options
python -m agentic_analyst.cli_agent \
  --csv your_data.csv \
  --output-dir custom_outputs \
  --model gpt-4
```

### Option 2: Docker (Recommended for Easy Setup)

```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Build and run with docker-compose (interactive mode)
docker-compose run --rm --build agentic-analyst
```

### 💬 Interactive Commands

- `/datasets` – View loaded datasets with schema information
- `/history` – See conversation history
- `/help` – Show help message
- `/exit` – Quit the session



## 🏗️ How It Works

This tool uses **OpenAI's function calling** with two specialized tools:

### 🔧 Available Tools

1. **`execute_python_code`** - Generates and runs pandas/matplotlib code for data analysis
   - Analyzes your question and dataset metadata
   - Writes custom Python code dynamically
   - Executes in sandboxed environment
   - Returns results, insights, and generated tables/charts

2. **`transform_output`** - Converts cached results to different formats without re-parsing CSVs
   - Transforms previous results
   - Saves charts in different formats
   - No re-execution needed

### Agent Flow

1. **User asks question** → Agent receives question + dataset metadata
2. **LLM decides** → Chooses appropriate tool(s) to use
3. **Tool execution** → Runs code or transforms output
4. **Results** → Sent back to LLM for interpretation
5. **Response** → Natural language answer with insights
6. **Real-time display** → See each step as it executes with syntax-highlighted code

## 🏗️ Architecture

### Core Components

| Module | Purpose |
|--------|---------|
| `cli_agent.py` | Interactive REPL with beautiful formatting |
| `agent.py` | OpenAI function calling orchestrator |
| `tools/code_executor.py` | Code execution tool - generates and runs Python |
| `tools/output_transformer.py` | Output transformation tool - converts cached results |
| `sandbox.py` | Safe code execution with timeout |
| `data_loader.py` | CSV loading and schema inference |
| `session_state.py` | Session management and history |
| `llm_client.py` | OpenAI client wrapper |


## 🔒 Transparency & Safety

✅ **Real-time progress** - See each step as it executes  
✅ **Code visibility** - All generated code is syntax-highlighted and displayed  
✅ **Sandboxed execution** - 30-second timeout protection  
✅ **Error recovery** - Enhanced feedback helps LLM learn and retry  
✅ **File safety** - Timestamped outputs prevent overwrites  
✅ **Session memory** - Last 5 interactions cached for context  



## 🎯 What Makes This Different

- **No templates** - Code is generated fresh for each question
- **Self-correcting** - Agent learns from errors and retries automatically
- **Multi-step reasoning** - Can execute multiple analyses in sequence
- **Context-aware** - Remembers previous questions and results
- **Production-ready** - Docker support, error handling, beautiful UX


# Limitations & Trade-offs
- **In-memory, file-based analytics**: The agent loads CSVs into pandas DataFrames in a single Python process. This works well up to tens of thousands of rows, but very large files (hundreds of MBs / millions of rows) would be better served by a database or query engine (e.g. Postgres, DuckDB, BigQuery) and/or SQL generation instead of pure in-memory pandas.

- **Local-only runtime**: All analysis is done locally in one container / process. There is no built-in support for distributed compute or remote warehouses; adding those would require a separate “SQL / warehouse tool” and connection management.

- **Non-hardened sandbox (imports allowed)**: For demo purposes, the sandbox currently allows `import` and access to common Python libraries. This is **not** a strong security boundary:
  - It does not prevent:
    - File system damage (e.g. deleting or overwriting files)
    - Network attacks (e.g. making HTTP requests)
    - Data exfiltration (e.g. reading secrets and sending them out)
    - Arbitrary code execution (any Python the LLM writes can run)
    - Resource exhaustion (while there is a timeout, there are no strict CPU/RAM limits)
  - A production-ready setup would need stronger isolation, e.g.:
    - Locked-down Docker containers or lightweight VMs
    - Strict import whitelisting
    - Process-level sandboxing with OS permissions, cgroups, and no network


- File saving behavior is still rough
The current implementation of file saving is functional but not very smart. In some cases, the agent may attempt to:
    - Save files directly without clear user intent, or
    - Save more artifacts than necessary (e.g. all intermediate results instead of just the final output)

