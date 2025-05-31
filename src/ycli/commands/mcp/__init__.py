import click

from .add import mcp_add
from .list import mcp_list
from .delete import mcp_delete

@click.group('mcp')
def mcp_group():
    """Manage MCP server configurations."""
    pass

# Register mcp subcommands
mcp_group.add_command(mcp_add)
mcp_group.add_command(mcp_list)
mcp_group.add_command(mcp_delete)
