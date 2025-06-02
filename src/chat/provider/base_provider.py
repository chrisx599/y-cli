from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from chat.models import Message, Chat

class BaseProvider(ABC):
    @abstractmethod
    async def call_chat_completions(self, messages: List[Message], chat: Optional[Chat] = None, system_prompt: Optional[str] = None, model_config: Optional[Dict] = None) -> Tuple[Message, Optional[str]]:
        """Get a chat response from the provider.
        
        Args:
            messages: List of Message objects
            system_prompt: Optional system prompt to add at the start
            model_config: Optional dictionary for model-specific configuration
            
        Returns:
            Message: The assistant's response message
            external_id: Optional external ID for the chat
            
        Raises:
            Exception: If API call fails
        """
        pass
