import os
import sys
import logging
import warnings

# Suppress all warnings and set environment variables BEFORE importing any third-party libraries
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTHONUNBUFFERED'] = '0'
# Disable logging from known noisy libraries
os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'ERROR'
os.environ['TELEMETRY_ENABLED'] = 'false'

# Configure root logger to ERROR level and redirect to stderr BEFORE any imports
logging.basicConfig(
    level=logging.ERROR,
    stream=sys.stderr,
    format='%(levelname)s: %(message)s',
    force=True
)

# Redirect stdout temporarily to suppress import-time logging from third-party libraries
# This is critical for MCP protocol which uses stdout for JSON-RPC communication
_original_stdout = sys.stdout
_null_stream = open(os.devnull, 'w')
sys.stdout = _null_stream  # Temporarily redirect stdout to /dev/null during imports

try:
    from fastmcp import FastMCP
    from dotenv import load_dotenv
    import asyncio
    from pathlib import Path
finally:
    # Restore stdout after imports
    sys.stdout = _original_stdout
    _null_stream.close()

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from src.utils import assemble_project_path
from src.logger import logger

# Load environment variables (suppressing any output)
load_dotenv(override=True)

# Disable logging from third-party libraries that output to stdout
# This is critical for MCP protocol which uses stdout for JSON-RPC communication
for lib_name in ['browser_use_agent', 'telemetry', 'dotenv', 'docket', 'FastMCP', 'mcp.server.lowlevel']:
    lib_logger = logging.getLogger(lib_name)
    lib_logger.setLevel(logging.CRITICAL)
    lib_logger.disabled = True
    lib_logger.propagate = False

# Configure our custom logger to output to stderr only
if hasattr(logger, 'handlers'):
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            # Check if handler outputs to stdout
            if hasattr(handler, 'stream') and handler.stream == sys.stdout:
                logger.removeHandler(handler)
            elif hasattr(handler.stream, 'fileno'):
                try:
                    if handler.stream.fileno() == sys.stdout.fileno():
                        logger.removeHandler(handler)
                except:
                    pass
    # Add stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    if hasattr(logger, 'formatter'):
        stderr_handler.setFormatter(logger.formatter)
    else:
        stderr_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    stderr_handler.setLevel(logging.ERROR)  # Only show errors in MCP server
    logger.addHandler(stderr_handler)

# Initialize FastMCP
mcp = FastMCP("LocalMCP")
_mcp_tools_namespace = {}

async def register_tool_from_script(script_info):
    """
    Register a tool from a script content.
    """

    name = script_info.get("name", "UnnamedTool")
    description = script_info.get("description", "No description provided.")
    script_content = script_info.get("script_content", "")

    if script_content.startswith('```python'):
        script_content = script_content.replace('```python', '')
    if script_content.endswith('```'):
        script_content = script_content.replace('```', '')

    try:
        exec(script_content, _mcp_tools_namespace)
    except Exception as e:
        logger.error(f"Error executing script for tool '{name}': {e}")
        return

    tool_function = _mcp_tools_namespace.get(name, None)
    if tool_function is None:
        logger.error(f"Tool function '{name}' not found in script content.")
        return
    else:
        mcp.tool(
            tool_function,
            name=name,
            description=description,
        )
        logger.info(f"Tool '{name}' registered successfully.")

async def register_tools(script_info_path):
    """
    Register tools from a JSON file containing script information.
    """
    import json

    try:
        with open(script_info_path, 'r') as f:
            script_info_list = json.load(f)

        for script_info in script_info_list:
            await register_tool_from_script(script_info)

    except FileNotFoundError:
        logger.info(f"Script info file not found: {script_info_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from script info file: {script_info_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while registering tools: {e}")

    logger.info("All tools registered successfully.")

    mcp_tools = await mcp.get_tools()
    logger.info(f"Registered tools: {', '.join([tool for tool in mcp_tools])}")

if __name__ == "__main__":
    script_info_path = assemble_project_path(os.path.join("src", "mcp", "local", "mcp_tools_registry.json"))
    asyncio.run(register_tools(script_info_path))
    mcp.run()