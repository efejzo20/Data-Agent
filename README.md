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



# How I Approached the Problem
I understood the core requirement as building an agent that you can talk to about data stored in structured formats, specifically CSV files. Since the data is already structured, I decided that the most robust way to work with it is through code (e.g., pandas), rather than using a RAG-style approach that would be better suited for unstructured formats like PDFs.

The first design decision was to use an LLM that can call tools. I needed at least one tool that could execute Python code against the loaded CSVs. That part was straightforward: the LLM generates code, and the tool runs it. The less obvious part was how to handle saving outputs, whether saving should be a separate tool or part of the same execution tool. I chose to keep it together in one tool that both executes the code and handles saving results. My reasoning was that, within a single run, the LLM is already “thinking” about the code and the files it produces, so combining execution and saving keeps the flow simpler. I still designed it so that this behaviour can be refactored or extended later.

Next, I implemented a sandbox for running the generated code. Initially, I restricted imports to avoid arbitrary libraries and limit what the LLM could do (for safety and control). Since this is a first-step implementation, I later relaxed this slightly to support the necessary imports, noting that in a future iteration we could harden the sandbox further and isolate execution more strictly.

For the agent execution loop, I assumed the LLM would sometimes make mistakes, so it needs a way to iterate and self-correct. I pass any errors from the sandbox back to the LLM and let it refine the code. I run this in a loop of up to seven iterations until a valid result is produced or the attempts are exhausted. Once the code runs successfully, I send the resulting data and metadata back to the LLM so it can generate a final, user-friendly answer.

I also added a separate output-transformation tool. This tool handles cases where the LLM has already generated data and stored it in memory, and the user later asks to export or transform it. For example, saving it, converting it to another file format, or applying translations, without re-parsing the original CSV.

On the UX side, I used the rich library to make the CLI output more readable. I show each step to the user: which tool the LLM used, which CSVs were involved, and whether there were errors. This helps make the agent’s behaviour transparent.

Finally, to make the project easy to run on different machines, I added a Docker setup so the environment and dependencies are consistent across systems.


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

This can definitely be improved by adding more explicit control over what gets saved and when, but due to time constraints I did not refine this behavior further in this iteration.

