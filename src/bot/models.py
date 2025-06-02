"""Bot configuration models."""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

DEFAULT_OPENROUTER_CONFIG = {
    "provider": {
        "sort": "throughput"
    }
}

DEFAULT_MCP_SERVER_CONFIG = ["todo"]

@dataclass
class BotConfig:
    name: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    api_type: Optional[str] = None
    model: str = ""
    print_speed: int = None
    description: Optional[str] = None
    openrouter_config: Optional[Dict] = None
    plan_model_config: Optional[Dict] = None  # New field for plan model configuration
    act_model_config: Optional[Dict] = None   # New field for act model configuration
    prompts: Optional[List[str]] = None
    plan_prompts: Optional[str] = None # New field for plan-specific prompt (single string)
    act_prompts: Optional[str] = None  # New field for act-specific prompt (single string)
    mcp_servers: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    custom_api_path: Optional[str] = None
    reasoning_effort: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'BotConfig':
        return cls(**data)

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
