from typing import Any, List, Dict, Optional, AsyncGenerator, Tuple
from .base_provider import BaseProvider
from .display_manager_mixin import DisplayManagerMixin
import json
from types import SimpleNamespace
import httpx
from chat.models import Message, Chat
from bot.models import BotConfig
from ..utils.message_utils import create_message

class DifyProvider(BaseProvider, DisplayManagerMixin):
    def __init__(self, bot_config: BotConfig):
        """Initialize Dify settings.

        Args:
            bot_config: Bot configuration containing API settings
        """
        DisplayManagerMixin.__init__(self)
        self.bot_config = bot_config
        # self.chat_endpoint is now determined dynamically in call_chat_completions

    def _prepare_headers(self, api_key: str) -> Dict[str, str]:
        """Prepare headers for API request."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_request_body(self, messages: List[Message], chat: Optional[Chat] = None, system_prompt: Optional[str] = None, model: str = None) -> Dict[str, Any]:
        """Prepare request body for Dify API."""
        # Get the last user message as query
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            raise ValueError("No user messages found")
        
        last_user_msg = user_messages[-1]
        query = last_user_msg.content if isinstance(last_user_msg.content, str) else " ".join(part.text for part in last_user_msg.content)

        # Get conversation history for context
        conversation_messages = messages[:-1]  # Exclude the last user message
        
        body = {
            "query": query,
            "response_mode": "streaming",
            "user": "user",
            "inputs": {},
        }

        # Add conversation ID if chat has external_id
        if chat and chat.external_id:
            body["conversation_id"] = chat.external_id
        
        # Add model if provided
        if model:
            body["model"] = model

        return body

    async def call_chat_completions(self, messages: List[Message], chat: Optional[Chat] = None, system_prompt: Optional[str] = None, model_config: Optional[Dict] = None) -> Tuple[Message, Optional[str]]:
        """Get a chat response from Dify.
        
        Args:
            messages: List of Message objects
            chat: Optional Chat object to maintain conversation context
            system_prompt: Optional system prompt to add at the start
            model_config: Optional dictionary for model-specific configuration
            
        Returns:
            Message: The assistant's response message
            
        Raises:
            Exception: If API call fails
        """
        if not self.display_manager:
            raise Exception("Display manager not set for streaming response")

        # Use model_config if provided, otherwise fallback to bot_config
        current_base_url = model_config.get("base_url", self.bot_config.base_url) if model_config else self.bot_config.base_url
        current_api_key = model_config.get("api_key", self.bot_config.api_key) if model_config else self.bot_config.api_key
        current_custom_api_path = model_config.get("custom_api_path", self.bot_config.custom_api_path) if model_config else self.bot_config.custom_api_path
        current_model = model_config.get("model", self.bot_config.model) if model_config else self.bot_config.model

        chat_endpoint = current_custom_api_path if current_custom_api_path else "/chat-messages"
        headers = self._prepare_headers(current_api_key)
        body = self._prepare_request_body(messages, chat, system_prompt, current_model)

        try:
            async with httpx.AsyncClient(
                base_url=current_base_url,
            ) as client:
                async with client.stream(
                    "POST",
                    chat_endpoint,
                    headers=headers,
                    json=body,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()

                    message_id = None
                    conversation_id = None

                    async def generate_chunks():
                        nonlocal message_id, conversation_id
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    event = data.get('event')

                                    if event == 'error':
                                        raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

                                    elif event == 'message':
                                        content = data.get('answer', '')
                                        if not message_id:
                                            message_id = data.get('message_id')
                                        if not conversation_id:
                                            conversation_id = data.get('conversation_id')

                                        chunk_data = SimpleNamespace(
                                            choices=[SimpleNamespace(
                                                delta=SimpleNamespace(
                                                    content=content,
                                                    reasoning_content=None
                                                )
                                            )],
                                            model=current_model, # Use current_model
                                            provider="dify"
                                        )
                                        yield chunk_data

                                except json.JSONDecodeError:
                                    continue

                    content_full, reasoning_content_full = await self.display_manager.stream_response(generate_chunks())

                    return create_message(
                        "assistant",
                        content_full,
                        id=message_id,
                        provider="dify",
                        model=current_model # Use current_model
                    ), conversation_id

        except httpx.HTTPError as e:
            raise Exception(f"HTTP error getting chat response: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calling Dify API: {str(e)}")
