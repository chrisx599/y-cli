import click

from mcp_server.service import McpServerConfigService
from mcp_server.repository import McpServerConfigRepository

@click.command('delete')
@click.argument('name')
def mcp_delete(name):
    """Delete an MCP server configuration."""
    from config import config
    # Initialize repository and service
    repository = McpServerConfigRepository(config['mcp_config_file'])
    service = McpServerConfigService(repository)
    
    # Delete the config
    if service.delete_config(name):
        click.echo(f"MCP server '{name}' deleted successfully")
    else:
        click.echo(f"MCP server '{name}' not found")
