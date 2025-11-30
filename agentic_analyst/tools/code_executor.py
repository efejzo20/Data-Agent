"""Code execution tool schema and implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..sandbox import SandboxResult, execute_code
from ..session_state import SessionState


class CodeExecutorInput(BaseModel):
    """Schema for the code executor tool."""

    code: str = Field(
        description="Python code to execute. "
        "You can import any standard library modules (pandas, numpy, matplotlib, etc.). "
        "Access datasets via 'datasets' dict (e.g., datasets['dataset_name']). "
        "Assign results to 'result_df' (DataFrame), 'fig' (Figure), or 'insight' (string). "
        "Example: import pandas as pd; df = datasets['name']; result_df = df.groupby('col').mean()"
    )
    datasets_needed: List[str] = Field(
        description="List of dataset names required for this code execution. "
        "You can specify multiple datasets if you need to join/merge them.",
        default_factory=list,
    )
    description: str = Field(
        description="Brief description of what this code does",
        default="Executing analysis code",
    )
    save_chart: bool = Field(
        description="Set to true when user asks for a chart/visualization: "
        "'show me a pie chart', 'give me a bar chart', 'create a plot', 'visualize', "
        "'save chart', 'export chart', 'download chart'. "
        "Charts are typically meant to be saved, so set to true for most chart requests. "
        "This saves the chart as PNG to the outputs directory.",
        default=False,
    )
    save_table: bool = Field(
        description="Set to true ONLY if user explicitly asks to SAVE/EXPORT/DOWNLOAD the data: "
        "'save as CSV', 'export to Excel', 'download the table', 'create a file', 'save this'. "
        "Do NOT set to true for simple questions like 'show me' or 'what are' - only when user wants a file. "
        "This saves the result table as CSV to the outputs directory.",
        default=False,
    )


class CodeExecutorTool:
    """Tool for executing Python code in a sandboxed environment."""

    name = "execute_python_code"
    description = (
        "Execute Python code to analyze CSV data. "
        "You can import standard libraries like pandas, numpy, matplotlib, etc. "
        "Use this tool to perform data analysis, filtering, aggregation, joins, calculations, and generate charts. "
        "The code runs in a sandbox with access to common data science libraries. "
        "Returns execution results including any tables, charts, or insights generated."
    )

    @staticmethod
    def get_schema() -> Dict[str, Any]:
        """Return OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": CodeExecutorTool.name,
                "description": CodeExecutorTool.description,
                "parameters": CodeExecutorInput.model_json_schema(),
            },
        }

    @staticmethod
    def execute(
        arguments: Dict[str, Any], session: SessionState
    ) -> Dict[str, Any]:
        """Execute the tool with validated arguments."""
        try:
            input_data = CodeExecutorInput(**arguments)
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid arguments: {e}",
            }

        # Validate datasets exist
        for ds_name in input_data.datasets_needed:
            if ds_name not in session.datasets:
                return {
                    "success": False,
                    "error": f"Dataset '{ds_name}' not found. Available: {list(session.datasets.keys())}",
                }

        # Execute code in sandbox
        result = execute_code(
            code=input_data.code,
            session=session,
            description=input_data.description,
        )

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "stdout": result.stdout,
            }

        # Build response
        response: Dict[str, Any] = {
            "success": True,
            "description": input_data.description,
        }

        # Handle tables
        if result.tables:
            table_summaries = []
            for name, df in result.tables.items():
                summary = {
                    "name": name,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "preview": df.head(10).to_dict(orient="records"),
                }
                table_summaries.append(summary)
                session.remember_results(f"last_table_{name}", df)
                session.remember_results("last_table", df)

                # Save if requested
                if input_data.save_table:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(session.config.output_dir) / f"{name}_{timestamp}.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_path, index=False)
                    summary["saved_to"] = str(output_path)

            response["tables"] = table_summaries

        # Handle figures
        if result.figures:
            figure_summaries = []
            for name, fig in result.figures.items():
                summary = {"name": name}
                session.remember_results(f"last_figure_{name}", fig)
                session.remember_results("last_figure", fig)

                # Save if requested
                if input_data.save_chart:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(session.config.output_dir) / f"{name}_{timestamp}.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(output_path, dpi=150, bbox_inches="tight")
                    summary["saved_to"] = str(output_path)

                figure_summaries.append(summary)

            response["figures"] = figure_summaries

        # Handle insight
        if result.insight:
            response["insight"] = result.insight

        # Include stdout if any
        if result.stdout:
            response["stdout"] = result.stdout

        return response


def get_all_tools() -> List[Dict[str, Any]]:
    """Return all available tool schemas for OpenAI."""
    from .output_transformer import OutputTransformerTool
    return [
        CodeExecutorTool.get_schema(),
        OutputTransformerTool.get_schema(),
    ]


def execute_tool(
    tool_name: str, arguments: Dict[str, Any], session: SessionState
) -> Dict[str, Any]:
    """Execute a tool by name."""
    if tool_name == CodeExecutorTool.name:
        return CodeExecutorTool.execute(arguments, session)
    elif tool_name == "transform_output":
        from .output_transformer import OutputTransformerTool
        return OutputTransformerTool.execute(arguments, session)
    else:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
        }

