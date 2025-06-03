from typing import List, Dict, Optional
from contextlib import AsyncExitStack
from types import SimpleNamespace
import re
import json

from chat.models import Chat, Message
from .repository import ChatRepository
from .service import ChatService
from ycli.display_manager import DisplayManager
from ycli.input_manager import InputManager
from mcp_server.mcp_manager import MCPManager
from prompt.preset import time_prompt
from util import generate_id
from .utils.tool_utils import contains_tool_use, split_content
from .utils.message_utils import create_message
from .provider.base_provider import BaseProvider
from bot import BotConfig
from config import prompt_service, mcp_service
from config import config
from loguru import logger

class ChatManager:
    def __init__(
        self,
        repository: ChatRepository,
        display_manager: DisplayManager,
        input_manager: InputManager,
        mcp_manager: MCPManager,
        provider: BaseProvider,
        bot_config: BotConfig,
        chat_id: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize chat manager with required components.

        Args:
            repository: Repository for chat persistence
            display_manager: Manager for display and UI
            input_manager: Manager for user input
            mcp_manager: Manager for MCP operations
            provider: Chat provider for interactions
            bot_config: Bot configuration
            chat_id: Optional ID of existing chat to load
            verbose: Whether to show verbose output
        """
        self.service = ChatService(repository)
        self.bot_config = bot_config
        self.display_manager = display_manager
        self.input_manager = input_manager
        self.mcp_manager = mcp_manager
        self.provider = provider
        self.verbose = verbose

        self.plan_model_config = bot_config.plan_model_config if bot_config.plan_model_config is not None else {}
        self.act_model_config = bot_config.act_model_config if bot_config.act_model_config is not None else {}
        self.plan_prompts = bot_config.plan_prompts # This will be Optional[str]
        self.act_prompts = bot_config.act_prompts   # This will be Optional[str]
        self.current_mode = "plan"  # Default to plan mode
        self.model = bot_config.model # Keep for backward compatibility if needed

        # Set up cross-manager references
        self.provider.set_display_manager(display_manager)

        # Initialize chat state
        self.current_chat: Optional[Chat] = None
        self.external_id: Optional[str] = None
        self.plan_messages: List[Message] = []
        self.act_messages: List[Message] = []
        # self.system_prompt will be built dynamically before each call
        self.chat_id: Optional[str] = None
        self.continue_exist = False

        # Generate new chat ID immediately
        if chat_id:
            self.chat_id = chat_id
            self.continue_exist = True
        else:
            self.chat_id = generate_id()

    def set_mode(self, mode: str):
        """Set the current operating mode (plan or act)."""
        if mode not in ["plan", "act"]:
            self.display_manager.print_error(f"Invalid mode: {mode}. Mode must be 'plan' or 'act'.")
            return
        self.current_mode = mode
        self.display_manager.console.print(f"[green]Switched to {self.current_mode.upper()} mode.[/green]")

    @property
    def messages(self) -> List[Message]:
        """Returns the message history for the current mode."""
        if self.current_mode == "plan":
            return self.plan_messages
        else:
            return self.act_messages

    async def _load_chat(self, chat_id: str):
        """Load an existing chat by ID"""
        existing_chat = await self.service.get_chat(chat_id)
        if not existing_chat:
            self.display_manager.print_error(f"Chat {chat_id} not found")
            raise ValueError(f"Chat {chat_id} not found")

        # When loading an existing chat, populate the plan_messages with the full history
        # and clear act_messages. This assumes we always start in plan mode when continuing.
        self.plan_messages = existing_chat.messages
        self.act_messages = [] # Clear act messages on load
        self.current_chat = existing_chat

        if self.verbose:
            logger.info(f"Loaded {len(self.plan_messages)} messages into plan_messages from chat {chat_id}")

    def get_user_confirmation(self, content: str, server_name: str = None, tool_name: str = None) -> bool:
        """Get user confirmation before executing tool use

        Args:
            content: The tool use content
            server_name: The MCP server name
            tool_name: The tool name being executed

        Returns:
            bool: True if confirmed, False otherwise
        """
        # Check if auto_confirm is enabled for this tool
        if server_name and tool_name and self.bot_config.mcp_servers:
            server_config = mcp_service.get_config(server_name)
            if server_config and hasattr(server_config, 'auto_confirm') and tool_name in server_config.auto_confirm:
                if self.verbose:
                    logger.info(f"Auto-confirming tool use for {server_name}/{tool_name}")
                return True

        # Otherwise proceed with normal confirmation
        self.display_manager.console.print("\n[yellow]Tool use detected in response:[/yellow]")
        while True:
            response = input("\nWould you like to proceed with tool execution? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            self.display_manager.console.print("[yellow]Please answer 'y' or 'n'[/yellow]")

    async def process_user_message(self, user_message: Message):
        self.messages.append(user_message)
        self.display_manager.display_message_panel(user_message, index=len(self.messages) - 1)

        # Determine which model config to use based on current mode
        current_model_config = self.plan_model_config if self.current_mode == "plan" else self.act_model_config
        
        # Build system prompt dynamically before each call
        system_prompt = await self._build_system_prompt()

        assistant_message, external_id = await self.provider.call_chat_completions(self.messages, self.current_chat, system_prompt, model_config=current_model_config)
        if external_id:
            self.external_id = external_id
        await self.process_assistant_message(assistant_message)
        await self.persist_chat()

    async def process_assistant_message(self, assistant_message: Message):
        """Process assistant response and handle tool use and mode switching recursively"""
        content = assistant_message.content

        # First, handle tool use if present
        if contains_tool_use(content):
            plain_content, tool_content = split_content(content)

            mcp_tool = self.mcp_manager.extract_mcp_tool_use(tool_content)
            if not mcp_tool:
                # If it contains tool use but not a valid MCP tool, just append and return
                self.messages.append(assistant_message)
                self.display_manager.display_message_panel(assistant_message, index=len(self.messages) - 1)
                return

            server_name, tool_name, arguments = mcp_tool
            assistant_message.server = server_name
            assistant_message.tool = tool_name
            assistant_message.arguments = arguments
            assistant_message.content = plain_content # Update content to plain content

            self.messages.append(assistant_message)
            self.display_manager.display_message_panel(assistant_message, index=len(self.messages) - 1)

            if not self.get_user_confirmation(tool_content, server_name, tool_name):
                no_exec_msg = "Tool execution cancelled by user."
                self.display_manager.console.print(f"\n[yellow]{no_exec_msg}[/yellow]")
                user_message = create_message("user", no_exec_msg)
                self.messages.append(user_message)
                return

            tool_results = await self.mcp_manager.execute_tool(server_name, tool_name, arguments)
            
            # After tool execution, the result needs to go to the plan agent.
            # So, switch to plan mode and then feed the tool results as a user message.
            self.display_manager.console.print(f"\n[blue]Tool Execution Result:[/blue]")
            self.display_manager.console.print(f"[blue]  Tool: {tool_name}[/blue]")
            self.display_manager.console.print(f"[blue]  Result: {tool_results}[/blue]")

            # Switch to plan mode
            self.set_mode("plan")

            # Create a user message for the plan agent with the tool results
            # This message will be appended to plan_messages because mode is now "plan"
            plan_agent_update = create_message("user", json.dumps({
                "action": "tool_execution_feedback", # New action for plan agent
                "tool_name": tool_name,
                "tool_arguments": arguments,
                "tool_results": tool_results
            }))
            await self.process_user_message(plan_agent_update) # Update plan agent
            return # Tool use handled, return

        # If no tool use, check for mode switching directives based on JSON output
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            # If not JSON, just append the message and return
            self.messages.append(assistant_message)
            self.display_manager.display_message_panel(assistant_message, index=len(self.messages) - 1)
            return

        if self.current_mode == "plan" and "execution_plan" in parsed_content:
            ongoing_step = None
            for step in parsed_content["execution_plan"]["steps"]:
                if step.get("status") == "ongoing":
                    ongoing_step = step
                    break

            if ongoing_step:
                action_description = ongoing_step.get("action_description", "No description")
                tool_info = ongoing_step.get("tool", {})
                tool_name = tool_info.get("name", "unknown_tool")
                tool_purpose = tool_info.get("purpose", "no purpose specified")
                input_requirements = tool_info.get("input_requirements", [])

                self.display_manager.console.print(f"\n[blue]Plan Agent: Identified Ongoing Step:[/blue]")
                self.display_manager.console.print(f"[blue]  Description: {action_description}[/blue]")
                self.display_manager.console.print(f"[blue]  Tool: {tool_name} ({tool_purpose})[/blue]")
                self.display_manager.console.print(f"[blue]  Input Requirements: {input_requirements}[/blue]")

                # Append the plan agent's full message to its history
                self.messages.append(assistant_message)
                self.display_manager.display_message_panel(assistant_message, index=len(self.messages) - 1)

                # Switch to act mode
                self.set_mode("act")

                # Create a new user message for the act agent with the extracted step details
                act_agent_instruction = create_message("user", json.dumps({
                    "action": "execute_step",
                    "description": action_description,
                    "tool": tool_info,
                    "input_requirements": input_requirements
                }))
                await self.process_user_message(act_agent_instruction) # Initiate act agent's turn
                return


        # If no tool use and no recognized JSON directives, just append the message
        self.messages.append(assistant_message)
        self.display_manager.display_message_panel(assistant_message, index=len(self.messages) - 1)

    async def persist_chat(self):
        """Persist current chat state"""
        # Combine plan and act messages for persistence
        all_messages = sorted(self.plan_messages + self.act_messages, key=lambda m: m.unix_timestamp)

        if not self.current_chat:
            # Create new chat with pre-generated ID
            self.current_chat = await self.service.create_chat(all_messages, self.external_id, self.chat_id)
        else:
            # Update existing chat - external_id will be preserved automatically
            self.current_chat = await self.service.update_chat(self.current_chat.id, all_messages, self.external_id)

    async def run(self):
        """Run the chat session"""
        async with AsyncExitStack() as exit_stack:
            try:
                if self.verbose:
                    logger.info("Starting chat session...")
                # if cloudflare repository, sync
                storage_type = config.get('storage_type', 'file')
                if storage_type == 'cloudflare':
                    await self.service.repository._sync_from_r2_if_needed()
                # Load chat if chat_id was provided and not already loaded
                if self.continue_exist:
                    await self._load_chat(self.chat_id)
                else:
                    pass
                    # await self.service.repository.get_chat(None)
                if self.verbose:
                    logger.info("Chat loaded successfully")

                # Initialize MCP servers once
                if self.bot_config.mcp_servers:
                    await self.mcp_manager.connect_to_servers(self.bot_config.mcp_servers)

                if self.verbose:
                    self.display_manager.display_help()

                # Display existing messages if continuing from a previous chat
                if self.messages:
                    self.display_manager.display_chat_history(self.messages)

                while True:
                    # Get user input, multi-line flag, and line count
                    user_input, is_multi_line, line_count = self.input_manager.get_input()

                    if self.input_manager.is_exit_command(user_input):
                        self.display_manager.console.print("\n[yellow]Goodbye![/yellow]")
                        break

                    # Handle mode switching commands
                    if user_input.lower() == "/plan":
                        self.set_mode("plan")
                        continue
                    if user_input.lower() == "/act":
                        self.set_mode("act")
                        continue

                    if not user_input:
                        self.display_manager.console.print("[yellow]Please enter a message.[/yellow]")
                        continue

                    # Handle copy command
                    if user_input.lower().startswith('copy '):
                        if self.input_manager.handle_copy_command(user_input, self.messages):
                            continue

                    # Add user message to history
                    user_message = create_message("user", user_input)
                    if is_multi_line:
                        # clear <<EOF line and EOF line
                        self.display_manager.clear_lines(2)

                    self.display_manager.clear_lines(line_count)

                    await self.process_user_message(user_message)

            except (KeyboardInterrupt, EOFError):
                self.display_manager.console.print("\n[yellow]Chat interrupted. Exiting...[/yellow]")
            finally:
                # Clear sessions on exit
                self.mcp_manager.clear_sessions()

    async def _build_system_prompt(self) -> str:
        """Builds the system prompt dynamically based on current mode and configurations."""
        system_prompt_parts = []

        # Determine the primary prompt based on current mode
        primary_prompt_name = None
        if self.current_mode == "plan" and self.plan_prompts:
            primary_prompt_name = self.plan_prompts
        elif self.current_mode == "act" and self.act_prompts:
            primary_prompt_name = self.act_prompts
        
        if primary_prompt_name:
            # If a mode-specific prompt is set, it completely overrides other prompts
            if primary_prompt_name not in ["mcp"]: # "mcp" is handled separately
                prompt_config = prompt_service.get_prompt(primary_prompt_name)
                if prompt_config:
                    system_prompt_parts.append(prompt_config.content)
        else:
            # Fallback to general prompts and default components if no mode-specific prompt is defined
            system_prompt_parts.append(time_prompt)

            if self.bot_config.prompts:
                for prompt_name in self.bot_config.prompts:
                    if prompt_name not in ["mcp"]: # "mcp" is handled separately
                        prompt_config = prompt_service.get_prompt(prompt_name)
                        if prompt_config:
                            system_prompt_parts.append(prompt_config.content)
        

        if self.bot_config.mcp_servers and self.current_mode == "act":
            await self.mcp_manager.connect_to_servers(self.bot_config.mcp_servers)
            mcp_prompt = await self.mcp_manager.get_mcp_prompt(self.bot_config.mcp_servers, prompt_service) + "\n"
            system_prompt_parts.append(mcp_prompt)
        
        return "\n".join(system_prompt_parts) + "\n"
