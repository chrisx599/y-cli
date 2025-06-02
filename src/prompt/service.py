"""Prompt configuration service."""

from typing import List, Optional
from .models import PromptConfig
from .repository import PromptRepository
from .mcp import mcp_prompt
from .preset import deep_research_prompt

class PromptService:
    """Service for managing prompt configurations.
    
    Provides business logic for prompt configuration management.
    """
    
    def __init__(self, repository: PromptRepository):
        """Initialize the prompt service with a repository.
        
        Args:
            repository: Repository for storing prompt configurations
        """
        self.repository = repository
        self._ensure_default_prompts()

    @property
    def _mcp_prompt_config(self) -> PromptConfig:
        """Get the mcp prompt configuration directly from code."""
        return PromptConfig(
            name="mcp",
            content=mcp_prompt,
            description="mcp prompt"
        )

    @property
    def _deep_research_prompt_config(self) -> PromptConfig:
        """Get the deep_research prompt configuration directly from code."""
        return PromptConfig(
            name="deep-research",
            content=deep_research_prompt,
            description="deep research prompt"
        )

    def _ensure_default_prompts(self) -> None:
        """Ensure default prompts (excluding MCP) exist in the repository."""
        # MCP prompt is now always loaded directly from code, not persisted.
        # Only ensure deep-research prompt is in the repository if not present.
        if not self.repository.get_config("deep-research"):
            self.repository.add_config(self._deep_research_prompt_config)

    def list_prompts(self) -> List[PromptConfig]:
        """List all prompt configurations, including the hardcoded MCP prompt.
        
        Returns:
            List of PromptConfig objects
        """
        prompts = self.repository.list_configs()
        # Ensure MCP prompt is always included and is the canonical version
        # If it exists in the repository, remove it to replace with the hardcoded one
        prompts = [p for p in prompts if p.name != "mcp"]
        prompts.insert(0, self._mcp_prompt_config) # Add MCP prompt at the beginning
        return prompts

    def get_prompt(self, name: str = "default") -> Optional[PromptConfig]:
        """Get a prompt configuration by name.
        
        Args:
            name: Name of the prompt to retrieve, defaults to "default"
            
        Returns:
            PromptConfig if found, None otherwise
        """
        if name == "mcp":
            return self._mcp_prompt_config
        return self.repository.get_config(name)

    def add_prompt(self, config: PromptConfig) -> PromptConfig:
        """Add a new prompt configuration or update existing one.
        
        Args:
            config: PromptConfig to add or update
            
        Returns:
            Added or updated PromptConfig
        """
        if config.name == "mcp":
            # Prevent adding/overwriting the hardcoded MCP prompt
            raise ValueError("Cannot add or modify the 'mcp' prompt as it is hardcoded.")
        return self.repository.add_config(config)

    def delete_prompt(self, name: str) -> bool:
        """Delete a prompt configuration by name.
        
        Args:
            name: Name of the prompt to delete
            
        Returns:
            True if deleted, False if not found or cannot delete (default)
        """
        if name == "mcp":
            # Prevent deleting the hardcoded MCP prompt
            return False # Cannot delete hardcoded prompt
        return self.repository.delete_config(name)
