"""OpenAI-based agent orchestrator using function calling."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .llm_client import get_client
from .session_state import SessionState
from .tools.code_executor import execute_tool, get_all_tools

console = Console()


class AgentError(Exception):
    """Raised when agent execution fails."""


def _classify_error(error_msg: str) -> str:
    """Classify the type of error for better feedback."""
    error_lower = error_msg.lower()
    
    if "not defined" in error_lower or "name" in error_lower and "is not defined" in error_lower:
        return "NameError - Variable not defined"
    elif "keyerror" in error_lower or "key" in error_lower:
        return "KeyError - Column or key not found"
    elif "attributeerror" in error_lower:
        return "AttributeError - Method or attribute doesn't exist"
    elif "typeerror" in error_lower:
        return "TypeError - Wrong data type used"
    elif "valueerror" in error_lower:
        return "ValueError - Invalid value"
    elif "indexerror" in error_lower:
        return "IndexError - Index out of range"
    elif "syntaxerror" in error_lower:
        return "SyntaxError - Invalid Python syntax"
    else:
        return "ExecutionError"


def _get_error_suggestion(error_msg: str) -> str:
    """Provide helpful suggestions based on the error."""
    error_lower = error_msg.lower()
    
    if "not defined" in error_lower:
        # Extract variable name if possible
        if "name '" in error_lower:
            var_name = error_msg.split("name '")[1].split("'")[0]
            return f"Variable '{var_name}' was used before being defined. Define it before using it, or check for typos."
        return "A variable was used before being defined. Make sure all variables are defined before use."
    
    elif "keyerror" in error_lower:
        return "A column or key doesn't exist in the DataFrame. Check the column names in the dataset metadata."
    
    elif "attributeerror" in error_lower:
        return "Trying to use a method or attribute that doesn't exist. Check the pandas/numpy documentation."
    
    elif "typeerror" in error_lower:
        return "Wrong data type used in an operation. Check that you're using compatible types (e.g., numeric operations on numeric columns)."
    
    elif "valueerror" in error_lower:
        return "Invalid value provided to a function. Check the function's expected input format."
    
    elif "syntaxerror" in error_lower:
        return "Python syntax error in your code. Review the code for missing colons, parentheses, or quotes."
    
    else:
        return "Review your code logic and check the error message for clues. Try a simpler approach."


def build_system_prompt(session: SessionState) -> str:
    """Build system prompt with dataset context."""
    dataset_info = []
    for name, dataset in session.datasets.items():
        # Build detailed column information
        col_details = []
        for col_name, summary in dataset.column_summaries.items():
            dtype = summary.dtype
            examples = summary.example_values[:3] if summary.example_values else []
            
            # Add stats if available
            stats_str = ""
            if summary.stats:
                if dtype == "numeric":
                    min_val = summary.stats.get('min', 'N/A')
                    max_val = summary.stats.get('max', 'N/A')
                    mean_val = summary.stats.get('mean', 'N/A')
                    if min_val != 'N/A':
                        stats_str = f" [min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}]"
                elif dtype == "categorical":
                    stats_str = f" [{summary.stats.get('unique', 'N/A')} unique values]"
            
            # Format examples
            examples_str = ""
            if examples:
                examples_str = f" (e.g., {', '.join(str(ex)[:50] for ex in examples[:2])})"
            
            col_details.append(f"    • {col_name} ({dtype}){stats_str}{examples_str}")
        
        col_text = "\n".join(col_details)
        dataset_info.append(
            f"Dataset: {name}\n"
            f"  Rows: {len(dataset.df)}\n"
            f"  Columns ({len(dataset.df.columns)}):\n{col_text}"
        )
    
    datasets_text = "\n\n".join(dataset_info)
    
    # Build join suggestions
    join_info = ""
    if session.join_suggestions:
        join_lines = []
        for (ds1, ds2), keys in session.join_suggestions.items():
            join_lines.append(f"  • {ds1} ↔ {ds2}: can join on {', '.join(keys)}")
        if join_lines:
            join_info = "\n\nPOSSIBLE JOINS BETWEEN DATASETS:\n" + "\n".join(join_lines)
    
    return f"""You are a data analysis assistant. You help users analyze CSV data by writing and executing Python code.

    AVAILABLE DATASETS WITH METADATA:
    {datasets_text}{join_info}

    IMPORTANT CODE EXECUTION RULES:
    1. You can import standard libraries (pandas, numpy, matplotlib, etc.) as needed

    2. Access datasets via the 'datasets' dict (e.g., datasets['dataset_name'])

    3. You can use MULTIPLE datasets in a single code execution - merge/join them as needed

    4. Assign results to 'result_df' (DataFrame), 'fig' (matplotlib figure), or 'insight' (string)

    5. Use the column metadata above to choose the right columns and understand data types

    6. Set save_table=true ONLY when user explicitly asks to SAVE/EXPORT/DOWNLOAD ("save as CSV", "export to Excel", "download")
    Do NOT set save_table=true for simple questions like "what are" or "show me" - just return the answer

    7. Set save_chart=true when user asks for visualizations (pie chart, bar chart, plot) - charts are typically saved

    CODE TEMPLATE:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = datasets['dataset_name']
    result_df = df.groupby('column').mean()
    insight = "Analysis complete"
    ```

    Always explain your findings clearly to the user. Be concise but thorough."""


def build_context_messages(session: SessionState, max_history: int = 5) -> List[Dict[str, str]]:
    """Build conversation history for context."""
    messages = []
    recent_turns = session.chat_history[-max_history:] if session.chat_history else []
    
    for turn in recent_turns:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.message})
        elif turn.role == "assistant":
            content = turn.message
            if turn.insights:
                content = turn.insights
            messages.append({"role": "assistant", "content": content})
    
    return messages


def run_agent(
    user_message: str,
    session: SessionState,
    model: str = "gpt-5",
    max_iterations: int = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the agent with OpenAI function calling.
    
    Returns:
        Dict with 'response' (final message), 'tool_calls' (list of tools used),
        and 'artifacts' (any saved files/charts).
    """
    client = get_client()
    if not client:
        raise AgentError(
            "OpenAI client not initialized. Set OPENAI_API_KEY in .env file."
        )
    
    # Build messages
    messages = [
        {"role": "system", "content": build_system_prompt(session)},
    ]
    messages.extend(build_context_messages(session))
    messages.append({"role": "user", "content": user_message})
    
    tools = get_all_tools()
    tool_calls_made = []
    artifacts = []
    
    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            raise AgentError(f"OpenAI API error: {e}")
        
        message = response.choices[0].message
        
        # Check if we're done
        if message.content and not message.tool_calls:
            return {
                "response": message.content,
                "tool_calls": tool_calls_made,
                "artifacts": artifacts,
            }
        
        # Handle tool calls
        if message.tool_calls:
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_idx, tool_call in enumerate(message.tool_calls, 1):
                function_name = tool_call.function.name
                step_num = len(tool_calls_made) + 1
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    if verbose:
                        console.print(f"[bold red]⚠️  Failed to parse tool arguments: {e}[/bold red]")
                    result = {"success": False, "error": f"Invalid JSON arguments: {e}"}
                else:
                    # Show step header in real-time
                    if verbose:
                        console.print(f"\n[bold yellow]{'─'*80}[/bold yellow]")
                        console.print(f"[bold yellow]STEP {step_num}[/bold yellow] [dim]({function_name})[/dim]")
                        console.print(f"[bold yellow]{'─'*80}[/bold yellow]")
                        
                        description = arguments.get("description", "Executing code")
                        datasets_needed = arguments.get("datasets_needed", [])
                        datasets_str = ", ".join(datasets_needed) if datasets_needed else "no datasets"
                        
                        console.print(f"[bold]🔧 Tool:[/bold] [cyan]{function_name}[/cyan]")
                        console.print(f"[bold]📝 Task:[/bold] {description}")
                        console.print(f"[bold]📂 Datasets:[/bold] [cyan]{datasets_str}[/cyan]")
                        
                        # Show code if present
                        if "code" in arguments:
                            syntax = Syntax(
                                arguments["code"],
                                "python",
                                theme="monokai",
                                line_numbers=True,
                                word_wrap=True,
                                indent_guides=True,
                            )
                            panel = Panel(
                                syntax,
                                title=f"🐍 Generated Code (Step {step_num})",
                                border_style="cyan",
                                padding=(0, 1),
                            )
                            console.print(panel)
                        
                        console.print("[dim]🔄 Executing...[/dim]")
                    
                    result = execute_tool(function_name, arguments, session)
                    
                    # Show result immediately
                    if verbose:
                        if result.get("success"):
                            console.print(f"[bold green]✅ Step {step_num} completed successfully[/bold green]")
                            
                            # Show what was generated
                            if "tables" in result and result["tables"]:
                                saved_tables = [t for t in result["tables"] if "saved_to" in t]
                                memory_tables = [t for t in result["tables"] if "saved_to" not in t]
                                
                                if memory_tables:
                                    console.print(f"  [dim]Generated {len(memory_tables)} table(s) in memory[/dim]")
                                if saved_tables:
                                    console.print(f"  [green]💾 Saved {len(saved_tables)} table(s) to disk[/green]")
                                    for table in saved_tables:
                                        console.print(f"     [cyan]→ {table['saved_to']}[/cyan]")
                            
                            if "figures" in result and result["figures"]:
                                saved_figures = [f for f in result["figures"] if "saved_to" in f]
                                memory_figures = [f for f in result["figures"] if "saved_to" not in f]
                                
                                if memory_figures:
                                    console.print(f"  [dim]Generated {len(memory_figures)} chart(s) in memory[/dim]")
                                if saved_figures:
                                    console.print(f"  [green]💾 Saved {len(saved_figures)} chart(s) to disk[/green]")
                                    for fig in saved_figures:
                                        console.print(f"     [cyan]→ {fig['saved_to']}[/cyan]")
                            
                            if "insight" in result and result["insight"]:
                                console.print(f"  [magenta]💡 {result['insight']}[/magenta]")
                        else:
                            error_msg = result.get("error", "Unknown error")
                            console.print(f"[bold red]❌ Step {step_num} failed: {error_msg}[/bold red]")
                            if step_num < max_iterations:
                                console.print(f"[yellow]🔄 Agent will retry with a different approach...[/yellow]")
                    
                    # Track artifacts
                    if result.get("success"):
                        if "tables" in result:
                            for table in result["tables"]:
                                if "saved_to" in table:
                                    artifacts.append({
                                        "type": "table",
                                        "path": table["saved_to"],
                                    })
                        if "figures" in result:
                            for figure in result["figures"]:
                                if "saved_to" in figure:
                                    artifacts.append({
                                        "type": "chart",
                                        "path": figure["saved_to"],
                                    })
                
                tool_calls_made.append({
                    "tool": function_name,
                    "arguments": arguments if 'arguments' in locals() else {},
                    "result": result,
                })
                
                # Add tool response with enhanced error feedback
                if result.get("success"):
                    tool_response = json.dumps(result, default=str)
                else:
                    # Enhanced error message for failed executions
                    error_msg = result.get("error", "Unknown error")
                    enhanced_feedback = {
                        "success": False,
                        "error": error_msg,
                        "error_type": _classify_error(error_msg),
                        "suggestion": _get_error_suggestion(error_msg),
                        "previous_attempts": len([tc for tc in tool_calls_made if not tc["result"].get("success", True)]),
                    }
                    if "stdout" in result:
                        enhanced_feedback["stdout"] = result["stdout"]
                    tool_response = json.dumps(enhanced_feedback, default=str)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response,
                })
        else:
            # No tool calls and no content - shouldn't happen
            break
    
    # Max iterations reached
    raise AgentError(
        f"Agent exceeded maximum iterations ({max_iterations}). "
        "The task may be too complex or the model is stuck in a loop."
    )

