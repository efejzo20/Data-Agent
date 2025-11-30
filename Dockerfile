FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agentic_analyst/ ./agentic_analyst/
COPY case_study_germany_sample.csv .
COPY case_study_germany_treatment_costs_sample.csv .

# Create outputs directory
RUN mkdir -p outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the CLI agent
ENTRYPOINT ["python", "-m", "agentic_analyst.cli_agent"]
CMD []

