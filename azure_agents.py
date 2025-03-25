import os
import inspect
import logging
from typing import (
    Optional, Any, List, Callable, TypeVar, Generic, Dict, Union, Type,
    Iterable, Protocol, runtime_checkable, Tuple, cast, get_type_hints, overload
)
from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import json
import requests  # Add this import for web search
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import tiktoken  # Add tiktoken for token counting

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# Import all the necessary components from the agents package
from agents import (
    Agent, 
    Runner, 
    ModelSettings,
    set_default_openai_client, 
    set_default_openai_api, 
    set_tracing_disabled,
    function_tool,
    FunctionTool,
    Tool,
    ComputerTool,
    WebSearchTool,
    FileSearchTool,
    Handoff,
    handoff,
    InputGuardrail,
    OutputGuardrail,
    input_guardrail,
    output_guardrail,
    HandoffInputData,
    AgentHooks,
    RunHooks,
    default_tool_error_function
)
from agents.run_context import RunContextWrapper, TContext
from openai.types.responses.web_search_tool_param import UserLocation

# Configure logging
logger = logging.getLogger("azure_agents")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load environment variables from .env file when module is imported
load_dotenv()

# Disable tracing to avoid errors
set_tracing_disabled(True)

# Base context class for Azure Agents
class BaseContext:
    """
    Base context class for Azure Agents with built-in functionality for conversation history,
    handoffs tracking, and metadata storage.
    
    This provides a standard way to maintain state between agent runs and
    track important events like handoffs between agents.
    
    Example:
    ```python
    # Create a simple context
    context = BaseContext()
    
    # Run an agent with the context
    with agent_context(context) as ctx:
        result = run(agent, "Hello", ctx)
    
    # Access the conversation history
    for query, response in context.get_history():
        print(f"Q: {query}\nA: {response}\n")
    ```
    
    You can also extend this class to add custom functionality:
    ```python
    class MyCustomContext(BaseContext):
        def __init__(self):
            super().__init__()
            self.custom_data = {}
            
        def add_custom_data(self, key, value):
            self.custom_data[key] = value
    ```
    """
    def __init__(self, max_tokens: int = 30000, tokenizer_name: str = "cl100k_base", 
                 auto_summarize: bool = True, summarization_model: Optional[str] = None):
        """
        Initialize a new context instance.
        
        Args:
            max_tokens: Maximum number of tokens to maintain before summarization (default: 30000)
            tokenizer_name: Name of the tokenizer to use for counting tokens (default: cl100k_base for GPT-4)
            auto_summarize: Whether to automatically summarize conversation history when token limit is approached
            summarization_model: Optional model name to use for summarization (defaults to same as agent)
        """
        self.history = []  # Conversation history (query, response) tuples
        self.handoffs = []  # Agent handoffs (from_agent, to_agent) tuples
        self.metadata = {}  # General purpose metadata storage
        self.tool_executions = []  # Track tool executions
        self.session_start = datetime.datetime.now()
        
        # Token management
        self.max_tokens = max_tokens
        self.current_token_count = 0
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.auto_summarize = auto_summarize
        self.summarization_model = summarization_model
        self.summaries = []  # Store historical summaries
    
    def add_to_history(self, query: str, response: str) -> None:
        """
        Add a query-response pair to the conversation history.
        
        Args:
            query: The user query/input
            response: The agent's response
        """
        timestamp = datetime.datetime.now()
        
        # Calculate tokens for this exchange
        query_tokens = len(self.tokenizer.encode(query))
        response_tokens = len(self.tokenizer.encode(response))
        exchange_tokens = query_tokens + response_tokens
        
        # Add to history in a format that's compatible with message contexts
        exchange = {
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "tokens": exchange_tokens,
            "role_format": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
        }
        self.history.append(exchange)
        
        # Log that we're adding to history
        logger.info(f"Added to conversation history: Q: '{query[:30]}...' A: '{response[:30]}...'")
        
        # Update token count
        self.current_token_count += exchange_tokens
        
        # Check if we need to summarize
        if self.auto_summarize and self.current_token_count > self.max_tokens * 0.8:
            self._summarize_history()
    
    def _summarize_history(self) -> None:
        """
        Summarize the conversation history when it gets too large.
        This preserves the most recent exchanges while summarizing older ones.
        """
        if not self.history or len(self.history) < 3:
            return  # Not enough history to summarize
        
        logger.info(f"Summarizing conversation history. Current token count: {self.current_token_count}")
        
        # Keep the most recent exchanges (about 20% of max tokens)
        tokens_to_keep = int(self.max_tokens * 0.2)
        recent_exchanges = []
        tokens_in_recent = 0
        
        # Work backwards to keep the most recent exchanges
        for exchange in reversed(self.history):
            if tokens_in_recent + exchange["tokens"] <= tokens_to_keep:
                recent_exchanges.insert(0, exchange)
                tokens_in_recent += exchange["tokens"]
            else:
                break
        
        # If no recent exchanges fit, keep at least the most recent one
        if not recent_exchanges and self.history:
            recent_exchanges = [self.history[-1]]
        
        # The rest needs to be summarized
        exchanges_to_summarize = self.history[:-len(recent_exchanges)] if recent_exchanges else self.history
        
        if not exchanges_to_summarize:
            return  # Nothing to summarize
            
        # Create summarization prompt
        conversation_text = ""
        for exchange in exchanges_to_summarize:
            conversation_text += f"User: {exchange['query']}\n"
            conversation_text += f"Assistant: {exchange['response']}\n\n"
        
        prompt = f"""
        Please provide a concise summary of the following conversation between a user and an AI assistant.
        Capture the main topics discussed, key questions asked, and important information provided.
        Focus on retaining factual content and context that would be important for continuing the conversation.
        
        CONVERSATION:
        {conversation_text}
        
        SUMMARY:
        """
        
        # Perform the summarization using the OpenAI API
        try:
            # Use the global Azure client
            if _azure_client is not None:
                response = _azure_client.chat.completions.create(
                    model=self.summarization_model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                    messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations. MAkE SURE TO KEEP ALL USEFULL STUFF IN THE SUMMARRY"},
                             {"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                summary_text = response.choices[0].message.content
                
                # Add the summary to our list of summaries
                summary_tokens = len(self.tokenizer.encode(summary_text))
                self.summaries.append({
                    "timestamp": datetime.datetime.now(),
                    "summary": summary_text,
                    "tokens": summary_tokens,
                    "exchanges_summarized": len(exchanges_to_summarize)
                })
                
                # Replace the history with summary + recent exchanges
                self.history = [{
                    "timestamp": datetime.datetime.now(),
                    "query": "Previous conversation summary",
                    "response": summary_text,
                    "tokens": summary_tokens,
                    "is_summary": True
                }] + recent_exchanges
                
                # Recalculate token count
                self.current_token_count = summary_tokens + tokens_in_recent
                
                logger.info(f"Conversation history summarized. New token count: {self.current_token_count}")
                
            else:
                logger.warning("Summarization attempted but Azure client is not initialized")
        except Exception as e:
            logger.error(f"Error summarizing conversation history: {e}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history.
        
        Returns:
            List of dicts containing timestamp, query, and response
        """
        return self.history
    
    def get_latest_exchange(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent conversation exchange.
        
        Returns:
            Dict with timestamp, query, and response, or None if history is empty
        """
        if self.history:
            return self.history[-1]
        return None
    
    def get_token_count(self) -> int:
        """
        Get the current token count for the conversation history.
        
        Returns:
            Current number of tokens in the conversation history
        """
        return self.current_token_count
    
    def get_summaries(self) -> List[Dict[str, Any]]:
        """
        Get all historical summaries that have been generated.
        
        Returns:
            List of summary records with timestamp and summary text
        """
        return self.summaries
    
    def force_summarize(self) -> bool:
        """
        Force summarization of conversation history regardless of token count.
        
        Returns:
            True if summarization was successful, False otherwise
        """
        try:
            self._summarize_history()
            return True
        except Exception as e:
            logger.error(f"Error during forced summarization: {e}")
            return False
    
    def add_handoff(self, from_agent: str, to_agent: str) -> None:
        """
        Record a handoff between agents.
        
        Args:
            from_agent: Name of the agent handing off
            to_agent: Name of the agent receiving the handoff
        """
        timestamp = datetime.datetime.now()
        self.handoffs.append({
            "timestamp": timestamp,
            "from_agent": from_agent,
            "to_agent": to_agent
        })
    
    def get_handoffs(self) -> List[Dict[str, Any]]:
        """
        Get all recorded handoffs.
        
        Returns:
            List of handoff events with timestamp, from_agent, and to_agent
        """
        return self.handoffs
    
    def record_tool_execution(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> None:
        """
        Record the execution of a tool.
        
        Args:
            tool_name: Name of the tool
            inputs: Tool input arguments
            result: Result returned by the tool
        """
        timestamp = datetime.datetime.now()
        self.tool_executions.append({
            "timestamp": timestamp,
            "tool_name": tool_name,
            "inputs": inputs,
            "result": result
        })
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Store arbitrary metadata in the context.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Retrieve metadata from the context.
        
        Args:
            key: Metadata key
            default: Default value if key doesn't exist
            
        Returns:
            The stored value or the default
        """
        return self.metadata.get(key, default)
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
    
    def get_session_duration(self) -> datetime.timedelta:
        """Get the duration of the current session."""
        return datetime.datetime.now() - self.session_start
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the context data to a JSON file.
        
        Args:
            filepath: Path where to save the file
        """
        # Convert data to serializable format
        data = {
            "history": [
                {
                    "timestamp": h["timestamp"].isoformat(),
                    "query": h["query"],
                    "response": h["response"],
                    "tokens": h.get("tokens", 0),
                    "is_summary": h.get("is_summary", False)
                } for h in self.history
            ],
            "summaries": [
                {
                    "timestamp": s["timestamp"].isoformat(),
                    "summary": s["summary"],
                    "tokens": s["tokens"],
                    "exchanges_summarized": s["exchanges_summarized"]
                } for s in self.summaries
            ],
            "handoffs": [
                {
                    "timestamp": h["timestamp"].isoformat(),
                    "from_agent": h["from_agent"],
                    "to_agent": h["to_agent"]
                } for h in self.handoffs
            ],
            "tool_executions": [
                {
                    "timestamp": t["timestamp"].isoformat(),
                    "tool_name": t["tool_name"],
                    "inputs": t["inputs"],
                    "result": str(t["result"])  # Convert result to string for safe serialization
                } for t in self.tool_executions
            ],
            "metadata": self.metadata,
            "session_start": self.session_start.isoformat(),
            "saved_at": datetime.datetime.now().isoformat(),
            "current_token_count": self.current_token_count,
            "max_tokens": self.max_tokens,
            "tokenizer_name": self.tokenizer_name
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BaseContext':
        """
        Load context data from a JSON file.
        
        Args:
            filepath: Path to the saved context file
            
        Returns:
            A new BaseContext instance with the loaded data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create context with the same settings
        context = cls(
            max_tokens=data.get("max_tokens", 30000),
            tokenizer_name=data.get("tokenizer_name", "cl100k_base"),
            auto_summarize=True
        )
        
        # Set the token count directly
        context.current_token_count = data.get("current_token_count", 0)
        
        # Parse ISO timestamps back to datetime
        for h in data.get("history", []):
            context.history.append({
                "timestamp": datetime.datetime.fromisoformat(h["timestamp"]),
                "query": h["query"],
                "response": h["response"],
                "tokens": h.get("tokens", 0),
                "is_summary": h.get("is_summary", False)
            })
        
        for s in data.get("summaries", []):
            context.summaries.append({
                "timestamp": datetime.datetime.fromisoformat(s["timestamp"]),
                "summary": s["summary"],
                "tokens": s["tokens"],
                "exchanges_summarized": s["exchanges_summarized"]
            })
            
        for h in data.get("handoffs", []):
            context.handoffs.append({
                "timestamp": datetime.datetime.fromisoformat(h["timestamp"]),
                "from_agent": h["from_agent"],
                "to_agent": h["to_agent"]
            })
            
        for t in data.get("tool_executions", []):
            context.tool_executions.append({
                "timestamp": datetime.datetime.fromisoformat(t["timestamp"]),
                "tool_name": t["tool_name"],
                "inputs": t["inputs"],
                "result": t["result"]  # Stored as string
            })
            
        context.metadata = data.get("metadata", {})
        context.session_start = datetime.datetime.fromisoformat(data.get("session_start", context.session_start.isoformat()))
        
        return context

# Custom RunHooks for handoff notification
class HandoffNotificationHooks(RunHooks):
    """Built-in hooks that notify when a handoff occurs."""
    
    async def on_handoff(self, context, from_agent, to_agent):
        """Called when a handoff occurs between agents."""
        # Clear visual separation
        print("\n" + "=" * 40)
        
        # Handoff notification with clear indication of which agent is taking over
        handoff_message = f"🔄 HANDOFF: '{from_agent.name}' is handing off to '{to_agent.name}'"
        print(handoff_message)
        
        # Add descriptive message about the specialized agent
        specialization = to_agent.handoff_description or f"specialized in {to_agent.name.lower()} tasks"
        print(f"👉 Now '{to_agent.name}' will handle your request ({specialization})")
        
        print("=" * 40 + "\n")
        
        # If the context has an add_handoff method, use it to track handoffs
        if hasattr(context.context, 'add_handoff'):
            # First check if it's our BaseContext (or subclass)
            if isinstance(context.context, BaseContext):
                # Use new format with context tracking
                context.context.add_handoff(from_agent.name, to_agent.name)
            else:
                # Traditional context with simple tuple tracking
                context.context.add_handoff(from_agent.name, to_agent.name)

# Initialize Azure OpenAI client
try:
    _azure_client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Set the Azure client as the default for the agents package
    set_default_openai_client(_azure_client, use_for_tracing=False)
    
    # Force the use of chat completions API instead of responses API
    set_default_openai_api("chat_completions")
    
    logger.info("Azure OpenAI client initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing Azure OpenAI client: {e}")
    # Don't fail import, as we might be setting client manually later
    _azure_client = None

# Re-export function_tool to make it available from this module
azure_function_tool = function_tool

# Type variable for context
T = TypeVar('T')
R = TypeVar('R')

# Define a protocol for functions that can be tools
@runtime_checkable
class ToolFunction(Protocol):
    """Protocol for functions that can be used as tools."""
    __call__: Callable[..., Any]
    __name__: str
    __doc__: Optional[str]

@dataclass
class ToolDefinition:
    """A simplified structure to define a tool"""
    function: Callable[..., Any]
    name: Optional[str] = None
    description: Optional[str] = None
    
# Context management
@contextmanager
def agent_context(context: Any):
    """
    Context manager for agent execution.
    
    Example:
    ```python
    with agent_context(MyContext()) as ctx:
        result = run(agent, "What can you do?", ctx)
    ```
    """
    try:
        yield context
    except Exception as e:
        logger.exception(f"Error in agent context: {e}")
        raise

class AzureAgent(Agent, Generic[T]):
    """
    A simplified Agent class that's pre-configured to work with Azure OpenAI.
    
    Example usage:
    ```python
    # Basic usage
    agent = AzureAgent("My Assistant", "You are a helpful assistant.")
    
    # With tools
    agent = AzureAgent("Tool Assistant", "You can use tools.", tools=[my_tool_func])
    
    # With web search
    agent = AzureAgent("Search Assistant", "You are a helpful assistant.", use_search=True)
    
    # With specific model
    agent = AzureAgent("GPT-4", "You are GPT-4.", model_name="gpt-4o")
    
    # With temperature control
    agent = AzureAgent("Creative Assistant", "Be creative.", temperature=0.8)
    ```
    """
    def __init__(
        self, 
        name: str, 
        instructions: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Union[Callable, Tool, ToolDefinition]]] = None,
        handoffs: Optional[List[Union[Agent[Any], Handoff[T]]]] = None,
        input_guardrails: Optional[List[InputGuardrail[T]]] = None,
        output_guardrails: Optional[List[OutputGuardrail[T]]] = None,
        context_type: Optional[Type[Any]] = None,
        use_search: bool = False,
        search_location: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize an Azure-powered Agent with a simpler interface.
        
        Args:
            name: The name of the agent
            instructions: The instructions/system prompt for the agent
            model_name: The Azure OpenAI deployment name (defaults to AZURE_OPENAI_DEPLOYMENT_NAME env var)
            temperature: Shorthand for setting temperature (0-1). Higher = more creative.
            tools: List of functions or tools to use (plain functions will be wrapped automatically)
            handoffs: List of agents or handoffs this agent can delegate to
            input_guardrails: List of input guardrails
            output_guardrails: List of output guardrails  
            context_type: Type hint for the context (for better IDE support)
            use_search: Whether to enable web search capability
            search_location: Optional location string for geographically relevant search results
            **kwargs: Any additional arguments to pass to the Agent constructor
        """
        # Get model from environment if not specified
        model = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Configure model settings if temperature is provided
        model_settings = None
        if temperature is not None:
            model_settings = ModelSettings(temperature=temperature)
        
        # Automatically wrap plain functions as tools if needed
        processed_tools = []
        
        # Add tools from the parameters
        if tools:
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    # Process tool definition
                    processed_func = function_tool(
                        tool.function,
                        name_override=tool.name,
                        description_override=tool.description
                    )
                    processed_tools.append(processed_func)
                elif isinstance(tool, Tool):
                    # It's already a Tool instance
                    processed_tools.append(tool)
                elif callable(tool) and not isinstance(tool, Tool):
                    # If it's a plain function, wrap it with function_tool
                    processed_tools.append(function_tool(tool))
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")
                
        # Add web search tool if requested
        if use_search:
            # Create a flexible web search tool
            search_tool = function_tool(
                lambda query: _web_search_function(query, search_location),
                name_override="web_search",
                description_override="Search for current information on a topic or question."
            )
            processed_tools.append(search_tool)
            
            # Add URL fetching tool to work with search results
            fetch_url_tool = function_tool(
                _fetch_url_function,
                name_override="fetch_url",
                description_override="Fetch and parse content from a specific URL."
            )
            processed_tools.append(fetch_url_tool)
        
        # Extract any special kwargs
        handoff_description = kwargs.pop("handoff_description", None)
        output_type = kwargs.pop("output_type", None)
        hooks = kwargs.pop("hooks", None)
                
        super().__init__(
            name=name,
            instructions=instructions,
            model=model,
            handoff_description=handoff_description,
            handoffs=handoffs or [],
            model_settings=model_settings or ModelSettings(),
            tools=processed_tools,
            input_guardrails=input_guardrails or [],
            output_guardrails=output_guardrails or [],
            output_type=output_type,
            hooks=hooks,
            **kwargs
        )
    
    @classmethod
    def builder(cls, name: str) -> 'AgentBuilder':
        """
        Create an agent builder for a more fluent interface.
        
        Example:
        ```python
        agent = AzureAgent.builder("My Assistant") \
            .with_instructions("You are a helpful assistant") \
            .with_tools([get_weather, calculate]) \
            .with_temperature(0.7) \
            .build()
        ```
        
        Args:
            name: The name of the agent
            
        Returns:
            An AgentBuilder instance
        """
        return AgentBuilder(name)
    
    def with_web_search(self, location: Optional[str] = None) -> 'AzureAgent[T]':
        """
        Add web search capability to this agent.
        
        Args:
            location: Optional location string for geographically relevant results, e.g., "New York, USA"
            
        Returns:
            Self with web search tools added
        """
        # Get the search and URL fetching tools
        search_tools = web_search_tool(location)
        
        # Create a new list with the web search tools added
        new_tools = list(self.tools) + search_tools
        
        # Clone the agent with the new tools
        return self.clone(tools=new_tools)
    
    def with_chain_of_thought(self) -> 'AzureAgent[T]':
        """
        Enhance the agent with chain-of-thought prompting.
        
        Returns:
            A new agent with enhanced instructions for chain-of-thought reasoning
        """
        cot_instructions = f"""
{self.instructions}

When solving problems or responding to complex queries, please think through your reasoning step by step:

1. First, understand what is being asked and identify key information
2. Break down complex problems into smaller parts
3. Think through each part methodically
4. Explain your reasoning process
5. Arrive at a well-reasoned conclusion

This step-by-step approach will help ensure your responses are thorough and accurate.
"""
        return self.clone(instructions=cot_instructions.strip())

class AgentBuilder:
    """
    Builder class for creating agents with a fluent interface.
    """
    def __init__(self, name: str):
        self.name = name
        self.instructions: Optional[str] = None
        self.model_name: Optional[str] = None
        self.temperature: Optional[float] = None
        self.tools: List[Union[Callable, Tool, ToolDefinition]] = []
        self.handoffs: List[Union[Agent[Any], Handoff[Any]]] = []
        self.input_guardrails: List[InputGuardrail[Any]] = []
        self.output_guardrails: List[OutputGuardrail[Any]] = []
        self.use_search: bool = False
        self.search_location: Optional[str] = None
        self.kwargs: Dict[str, Any] = {}
    
    def with_instructions(self, instructions: str) -> 'AgentBuilder':
        """Set the agent's instructions"""
        self.instructions = instructions
        return self
    
    def with_model(self, model_name: str) -> 'AgentBuilder':
        """Set the model name/deployment name"""
        self.model_name = model_name
        return self
    
    def with_temperature(self, temperature: float) -> 'AgentBuilder':
        """Set the temperature for responses"""
        self.temperature = temperature
        return self
    
    def with_tools(self, tools: List[Union[Callable, Tool, ToolDefinition]]) -> 'AgentBuilder':
        """Add tools to the agent"""
        self.tools.extend(tools)
        return self
    
    def with_tool(self, tool: Union[Callable, Tool, ToolDefinition]) -> 'AgentBuilder':
        """Add a single tool to the agent"""
        self.tools.append(tool)
        return self
    
    def with_web_search(self, location: Optional[str] = None) -> 'AgentBuilder':
        """Add web search capability"""
        self.use_search = True
        self.search_location = location
        return self
    
    def with_handoffs(self, handoffs: List[Union[Agent[Any], Handoff[Any]]]) -> 'AgentBuilder':
        """Add handoffs to the agent"""
        self.handoffs.extend(handoffs)
        return self
    
    def with_handoff(self, handoff: Union[Agent[Any], Handoff[Any]]) -> 'AgentBuilder':
        """Add a single handoff to the agent"""
        self.handoffs.append(handoff)
        return self
    
    def with_input_guardrails(self, guardrails: List[InputGuardrail[Any]]) -> 'AgentBuilder':
        """Add input guardrails to the agent"""
        self.input_guardrails.extend(guardrails)
        return self
    
    def with_output_guardrails(self, guardrails: List[OutputGuardrail[Any]]) -> 'AgentBuilder':
        """Add output guardrails to the agent"""
        self.output_guardrails.extend(guardrails)
        return self
    
    def with_chain_of_thought(self) -> 'AgentBuilder':
        """Enable chain-of-thought prompting"""
        if not self.instructions:
            raise ValueError("Instructions must be set before enabling chain-of-thought")
            
        cot_instructions = f"""
{self.instructions}

When solving problems or responding to complex queries, please think through your reasoning step by step:

1. First, understand what is being asked and identify key information
2. Break down complex problems into smaller parts
3. Think through each part methodically
4. Explain your reasoning process
5. Arrive at a well-reasoned conclusion

This step-by-step approach will help ensure your responses are thorough and accurate.
"""
        self.instructions = cot_instructions.strip()
        return self
    
    def with_option(self, key: str, value: Any) -> 'AgentBuilder':
        """Set any additional option for the agent constructor"""
        self.kwargs[key] = value
        return self
    
    def build(self) -> AzureAgent[Any]:
        """Build the agent with the configured options"""
        if not self.instructions:
            raise ValueError("Instructions must be set using with_instructions()")
            
        return AzureAgent(
            name=self.name,
            instructions=self.instructions,
            model_name=self.model_name,
            temperature=self.temperature,
            tools=self.tools,
            handoffs=self.handoffs,
            input_guardrails=self.input_guardrails,
            output_guardrails=self.output_guardrails,
            use_search=self.use_search,
            search_location=self.search_location,
            **self.kwargs
        )

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """
    Decorator to create a tool from a function.
    This is a simpler version of function_tool.
    
    Example:
    ```python
    @tool(name="get_weather", description="Get weather for a location")
    def get_weather(location: str) -> dict:
        # Function implementation
        return {"temperature": 72, "conditions": "sunny"}
    ```
    
    Args:
        name: Optional override for the tool name (defaults to function name)
        description: Optional override for the tool description (defaults to function docstring)
        
    Returns:
        A decorator that creates a FunctionTool from the decorated function
    """
    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return function_tool(
            func,
            name_override=name,
            description_override=description
        )
    return decorator

def define_tool(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None
) -> ToolDefinition:
    """
    Create a tool definition from a function.
    
    Args:
        func: The function to use as a tool
        name: Optional override for the tool name (defaults to function name)
        description: Optional override for the tool description (defaults to function docstring)
        
    Returns:
        A ToolDefinition that can be used when creating an agent
    """
    return ToolDefinition(
        function=func,
        name=name,
        description=description
    )

def azure_handoff(
    agent: Union[Agent[T], Callable[[RunContextWrapper[Any], str], Agent[T]]],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
) -> Handoff[T]:
    """
    Create a simpler handoff configured for Azure OpenAI.
    
    Args:
        agent: The agent to handoff to
        tool_name: Optional custom name for the handoff tool
        tool_description: Optional custom description for the handoff tool
        
    Returns:
        A Handoff object
    """
    # Store the description for use in handoff notification
    if isinstance(agent, Agent):
        agent.handoff_description = tool_description
        
    return handoff(
        agent=agent,
        tool_name_override=tool_name,
        tool_description_override=tool_description
    )

# Custom handlers for common tool types to simplify creation
@tool(name="get_current_time", description="Returns the current date and time")
def datetime_tool() -> str:
    """
    Get the current date and time.
    
    Returns:
        A string with the current date and time in format YYYY-MM-DD HH:MM:SS
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Keep a factory function for backward compatibility
def get_datetime_tool() -> FunctionTool:
    """
    Legacy function to create a datetime tool (kept for backward compatibility).
    Please use the datetime_tool function directly instead.
    
    Returns:
        A tool that returns the current date and time
    """
    return function_tool(
        datetime_tool,
        name_override="get_current_time",
        description_override="Returns the current date and time."
    )

def web_search_tool(location: Optional[str] = None) -> List[FunctionTool]:
    """
    Create web search tools (search and URL fetching).
    
    Args:
        location: Optional location string for geographically relevant results
        
    Returns:
        A list of FunctionTools for web search capabilities
    """
    search_func = lambda query: _web_search_function(query, location)
    
    search_tool = function_tool(
        search_func,
        name_override="web_search",
        description_override="Search the web for information."
    )
    
    fetch_url_tool = function_tool(
        _fetch_url_function,
        name_override="fetch_url",
        description_override="Fetch and parse content from a specific URL."
    )
    
    return [search_tool, fetch_url_tool]

def run(agent: Agent, user_input: str, context: Any = None) -> Any:
    """
    Run an agent with Azure OpenAI (synchronously).
    
    This is a simplified version of run_sync that optionally takes a context.
    
    Args:
        agent: The agent to run
        user_input: The user input to send to the agent
        context: Optional context object to pass to the agent
        
    Returns:
        The result of running the agent
    """
    # Simple solution: add conversation history directly to the prompt
    if context is not None and hasattr(context, 'history') and context.history:
        # Check if we should summarize first (either based on tokens or number of exchanges)
        token_limit = 25000  # Conservative limit to avoid hitting context windows
        total_tokens = 0
        
        # Calculate total tokens from history
        for entry in context.history:
            if isinstance(entry, dict) and 'tokens' in entry:
                total_tokens += entry.get('tokens', 0)
        
        # If we're approaching the token limit or have too many exchanges, consider summarizing
        if total_tokens > token_limit or len(context.history) > 10:
            logger.info(f"Large conversation detected: {total_tokens} tokens, {len(context.history)} exchanges")
            
            # If context has summarization method, use it
            if hasattr(context, '_summarize_history') and callable(getattr(context, '_summarize_history')):
                logger.info("Summarizing conversation history before continuing")
                context._summarize_history()
        
        # Build conversation history as markdown
        conversation_summary = "## CONVERSATION HISTORY (ALWAYS REFERENCE THIS IN YOUR RESPONSE):\n\n"
        
        # Check if there are summaries to include
        if hasattr(context, 'summaries') and context.summaries:
            # Include the most recent summary
            latest_summary = context.summaries[-1]
            conversation_summary += f"**Summary of Previous Conversation**: {latest_summary.get('summary', 'Previous conversation available.')}\n\n"
            conversation_summary += "---\n\n"
        
        # Include conversation entries
        for i, entry in enumerate(context.history):
            if isinstance(entry, dict):
                # Skip entries that are marked as summaries
                if entry.get('is_summary', False):
                    conversation_summary += f"**Previous Conversation Summary**: {entry.get('response', '')}\n\n"
                    conversation_summary += "---\n\n"
                    continue
                    
                conversation_summary += f"**User**: {entry.get('query', '')}\n\n"
                conversation_summary += f"**Assistant**: {entry.get('response', '')}\n\n"
                conversation_summary += "---\n\n"
            elif isinstance(entry, tuple) and len(entry) == 2:
                conversation_summary += f"**User**: {entry[0]}\n\n"
                conversation_summary += f"**Assistant**: {entry[1]}\n\n"
                conversation_summary += "---\n\n"
                
        # Add clear instruction to reference the history
        conversation_summary += "## CURRENT QUERY (RESPOND TO THIS WHILE REFERENCING INFORMATION FROM THE HISTORY ABOVE):\n\n"
        
        # Modify user input to include the conversation history
        enhanced_input = f"{conversation_summary}{user_input}"
        
        # Log what we're doing
        logger.info(f"Enhanced query with conversation history ({len(context.history)} previous exchanges)")
        
        # Run with the enhanced input
        kwargs = {}
        # Add handoff notification hooks - still need this for handoffs to work
        hooks = HandoffNotificationHooks()
        kwargs["hooks"] = hooks
        
        # Run with the enhanced input while preserving hooks
        result = Runner.run_sync(agent, enhanced_input, **kwargs)
        
        # Update the context with the interaction
        if hasattr(context, 'add_to_history'):
            context.add_to_history(user_input, result.final_output)
            
        return result
    else:
        # No history, just run normally
        kwargs = {}
        # Add handoff notification hooks - still need this for handoffs to work
        hooks = HandoffNotificationHooks()
        kwargs["hooks"] = hooks
        
        result = Runner.run_sync(agent, user_input, **kwargs)
        
        # Update the context if available
        if context is not None and hasattr(context, 'add_to_history'):
            context.add_to_history(user_input, result.final_output)
            
        return result

# For backwards compatibility
run_sync = run
run_async = Runner.run

# Helper function to create a simple agent with tools
def create_tool_agent(name: str, instructions: str, tools: List[Union[Callable, Tool, ToolDefinition]], **kwargs) -> AzureAgent:
    """
    Create an agent with the specified tools.
    
    Args:
        name: The name of the agent
        instructions: The instructions for the agent
        tools: List of tool functions or tool instances
        **kwargs: Additional arguments to pass to AzureAgent
        
    Returns:
        An AzureAgent with the specified tools
    """
    return AzureAgent(name=name, instructions=instructions, tools=tools, **kwargs)

# Helper function to create an agent with a specific temperature
def create_creative_agent(name: str, instructions: str, temperature: float = 0.7, **kwargs) -> AzureAgent:
    """
    Create an agent with higher temperature for more creative responses.
    
    Args:
        name: The name of the agent
        instructions: The instructions for the agent
        temperature: Temperature value (0-1), defaults to 0.7
        **kwargs: Additional arguments to pass to AzureAgent
        
    Returns:
        An AzureAgent with the specified temperature
    """
    return AzureAgent(name=name, instructions=instructions, temperature=temperature, **kwargs)

# Helper to set verbosity level
def set_verbosity(level: int) -> None:
    """
    Set the verbosity level for logging.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
    """
    logger.setLevel(level)

# Function to connect to Azure OpenAI with custom parameters
def connect_to_azure(
    api_key: str,
    api_version: str,
    endpoint: str,
    deployment_name: Optional[str] = None
) -> None:
    """
    Connect to Azure OpenAI with custom parameters.
    
    Args:
        api_key: Azure OpenAI API key
        api_version: Azure OpenAI API version
        endpoint: Azure OpenAI endpoint
        deployment_name: Optional default deployment name
    """
    global _azure_client
    
    # Initialize Azure OpenAI client
    _azure_client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    
    # Set the Azure client as the default for the agents package
    set_default_openai_client(_azure_client, use_for_tracing=False)
    
    # Force the use of chat completions API instead of responses API
    set_default_openai_api("chat_completions")
    
    # Set the deployment name as an environment variable
    if deployment_name:
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = deployment_name
    
    logger.info("Azure OpenAI client initialized successfully with custom parameters")

# Define a custom web search function tool since WebSearchTool is not supported with Azure ChatCompletions
def _web_search_function(query: str, location: Optional[str] = None, num_results: int = 5) -> Dict[str, Any]:
    """
    Performs a real web search using multiple fallback options.
    
    Args:
        query: The search query
        location: Optional location for geographically relevant results
        num_results: Number of results to return (default 5)
        
    Returns:
        Dictionary with actual search results
    """
    try:
        # Prepare the search query
        search_query = query.strip()
        if location and location.lower() != "global":
            search_query = f"{search_query} {location}"
            
        logger.info(f"Performing web search for: {search_query}")
        
        # Initialize results
        results = []
        
        # Try multiple search engines in sequence until one works
        search_engines = [
            # DuckDuckGo only
            {
                "name": "DuckDuckGo",
                "url": f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Referer": "https://duckduckgo.com/",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                },
                "result_selector": [".result", ".web-result", ".links_main", ".nrn-react-div"],
                "title_selector": [".result__title", "h2", ".link", "a", ".result__a"],
                "url_selector": [".result__url", "a.result__a", "a", ".result__a"],
                "snippet_selector": [".result__snippet", ".snippet", ".result__snippet"]
            }
        ]
        
        success = False
        
        # Try each search engine until one returns results
        for engine in search_engines:
            if success:
                break
                
            try:
                logger.info(f"Trying search with {engine['name']}...")
                
                # Fetch search results
                response = requests.get(
                    engine["url"], 
                    headers=engine["headers"],
                    timeout=15
                )
                
                if response.status_code == 200:
                    # Parse response
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Handle multiple possible selectors
                    result_selector = engine["result_selector"]
                    result_elements = []
                    
                    if isinstance(result_selector, list):
                        for selector in result_selector:
                            result_elements = soup.select(selector)
                            if result_elements:
                                break
                    else:
                        result_elements = soup.select(result_selector)
                    
                    # Limit the number of results
                    result_elements = result_elements[:num_results]
                    
                    # Process results
                    for result in result_elements:
                        # Extract title
                        title = "No title found"
                        for title_selector in engine["title_selector"] if isinstance(engine["title_selector"], list) else [engine["title_selector"]]:
                            title_el = result.select_one(title_selector)
                            if title_el:
                                title = title_el.get_text().strip()
                                break
                        
                        # Extract URL
                        url = "#"
                        for url_selector in engine["url_selector"] if isinstance(engine["url_selector"], list) else [engine["url_selector"]]:
                            url_el = result.select_one(url_selector)
                            if url_el:
                                if url_el.name == 'a' and url_el.has_attr('href'):
                                    url = url_el['href']
                                    # Clean up redirects
                                    if url.startswith('/url?') or url.startswith('/search?'):
                                        import re
                                        match = re.search(r'[?&]q=([^&]+)', url)
                                        if match:
                                            url = match.group(1)
                                    # Clean up relative URLs
                                    if url.startswith('/'):
                                        url = f"https://{engine['name'].lower()}.com{url}"
                                elif url_el.name == 'cite':
                                    url = url_el.get_text().strip()
                                break
                        
                        # Extract snippet
                        snippet = "No description available"
                        for snippet_selector in engine["snippet_selector"] if isinstance(engine["snippet_selector"], list) else [engine["snippet_selector"]]:
                            snippet_el = result.select_one(snippet_selector)
                            if snippet_el:
                                snippet = snippet_el.get_text().strip()
                                break
                        
                        results.append({
                            "title": title,
                            "snippet": snippet,
                            "url": url,
                            "source": engine["name"]
                        })
                    
                    if results:
                        logger.info(f"Found {len(results)} search results from {engine['name']}")
                        success = True
                        break
                    else:
                        logger.warning(f"No results found in {engine['name']} response")
                else:
                    logger.warning(f"{engine['name']} returned status code {response.status_code}")
            
            except Exception as e:
                logger.error(f"Error with {engine['name']} search: {e}")
                # Continue to next engine
        
        # If any results were found
        if results:
            return {
                "search_query": search_query,
                "results": results,
                "instructions": "You can fetch content from any of these URLs by using the fetch_url function. Example: fetch_url(url)"
            }
        else:
            # Fallback to direct knowledge if no search engine worked
            # Try to get some basic information from Wikipedia API as last resort
            try:
                logger.info("Attempting Wikipedia fallback search...")
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(search_query)}&limit=5&namespace=0&format=json"
                wiki_response = requests.get(wiki_url, timeout=10)
                
                if wiki_response.status_code == 200:
                    wiki_data = wiki_response.json()
                    if len(wiki_data) >= 4:
                        titles = wiki_data[1]
                        descriptions = wiki_data[2]
                        urls = wiki_data[3]
                        
                        for i in range(min(len(titles), 5)):
                            results.append({
                                "title": titles[i],
                                "snippet": descriptions[i] if i < len(descriptions) else "Wikipedia article",
                                "url": urls[i] if i < len(urls) else "#",
                                "source": "Wikipedia API"
                            })
                        
                        if results:
                            logger.info(f"Found {len(results)} results from Wikipedia API")
                            return {
                                "search_query": search_query,
                                "results": results,
                                "instructions": "You can fetch content from any of these URLs by using the fetch_url function. Example: fetch_url(url)"
                            }
            except Exception as e:
                logger.error(f"Error with Wikipedia fallback: {e}")
            
            # If all attempts failed
            return {
                "search_query": search_query,
                "message": "Search failed across multiple search engines. Please try a different query or approach.",
                "fallback": "I couldn't find current information on this topic. Let me answer based on what I know."
            }
                
    except Exception as e:
        logger.error(f"Error in web search function: {e}")
        return {
            "error": str(e),
            "search_query": query,
            "fallback": "I encountered an error while searching. Let me answer based on what I know."
        }

def _fetch_url_function(url: str) -> Dict[str, Any]:
    """
    Fetches and parses the content of a specific URL.
    
    Args:
        url: The URL to fetch
        
    Returns:
        Dictionary with parsed content and metadata
    """
    try:
        logger.info(f"Fetching content from URL: {url}")
        
        # Clean up the URL if needed
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url.lstrip('/')
            
        # Set headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Add timeout to prevent hanging
        response = requests.get(url, headers=headers, timeout=15)
        
        # Check for successful response
        if response.status_code == 200:
            logger.info(f"Successfully received response from {url}")
            
            # Get content type
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Handle different content types
            if 'application/json' in content_type:
                # JSON content
                try:
                    json_data = response.json()
                    return {
                        "url": url,
                        "content_type": "json",
                        "content": str(json_data)[:4000],  # Limit length
                        "title": "JSON Content",
                        "message": "Successfully retrieved JSON data."
                    }
                except Exception as e:
                    logger.error(f"Error parsing JSON from {url}: {e}")
            
            elif 'text/plain' in content_type:
                # Plain text content
                text_content = response.text[:4000]  # Limit length
                return {
                    "url": url,
                    "content_type": "text",
                    "content": text_content,
                    "title": url.split('/')[-1] or "Text Content",
                    "message": "Successfully retrieved text content."
                }
                
            else:
                # Assume HTML content
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header']):
                    element.extract()
                
                # Get page title
                title = soup.title.string if soup.title else url.split('/')[-1] or "No title found"
                
                # Try to identify the main content
                main_content = None
                
                # Try different common content selectors
                content_selectors = [
                    'article', 'main', '.article', '.post', '.content', '#content', 
                    '.main-content', '#main-content', '.entry-content', '.post-content',
                    '[role="main"]', '.body', '#body'
                ]
                
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        main_content = content.get_text(separator='\n').strip()
                        break
                
                # If no main content container found, look for <p> tags in the body
                if not main_content:
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        main_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
                
                # If still no content, just get all text
                if not main_content or len(main_content) < 100:
                    main_content = soup.get_text(separator='\n').strip()
                
                # Clean up the content
                import re
                
                # Remove excessive whitespace/newlines
                main_content = re.sub(r'\n\s*\n', '\n\n', main_content)
                main_content = re.sub(r'\s{2,}', ' ', main_content)
                
                # Find and clean up paragraphs for better readability
                main_content = re.sub(r'([.!?])\s*\n', r'\1\n\n', main_content)
                
                # Limit content length
                max_length = 4000
                if len(main_content) > max_length:
                    main_content = main_content[:max_length] + "...\n[Content truncated due to length]"
                
                # Extract meta description if available
                meta_desc = ""
                meta_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                if meta_tag and meta_tag.has_attr('content'):
                    meta_desc = meta_tag['content']
                
                # Extract all links for potential further exploration
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text().strip()
                    
                    # Filter out empty links, javascript, etc.
                    if (href and 
                        not href.startswith(('javascript:', '#', 'mailto:')) and 
                        link_text and len(link_text) > 1):
                        
                        # Make relative URLs absolute
                        if not href.startswith(('http://', 'https://')):
                            from urllib.parse import urljoin
                            href = urljoin(url, href)
                            
                        # Only include if not already in list and not the same as current URL
                        if href != url and not any(l['url'] == href for l in links):
                            links.append({
                                "url": href,
                                "text": link_text[:100]  # Limit text length
                            })
                
                # Limit number of links
                links = links[:10]
                
                # Create summary info
                summary = meta_desc
                if not summary and main_content:
                    # Create a summary from the first paragraph if no meta description
                    first_para = re.split(r'\n\s*\n', main_content)[0]
                    summary = first_para[:200] + ('...' if len(first_para) > 200 else '')
                
                return {
                    "url": url,
                    "title": title,
                    "content": main_content,
                    "summary": summary,
                    "links": links,
                    "instructions": "You can fetch content from any of these links by using the fetch_url function again."
                }
        
        elif response.status_code == 403:
            # Special handling for Forbidden errors
            return {
                "error": "Access to this page is forbidden (403 status). This website may have anti-scraping measures.",
                "url": url,
                "suggestion": "Try visiting the website directly in a browser or try a different source."
            }
            
        elif response.status_code == 404:
            # Special handling for Not Found errors
            return {
                "error": "The requested page was not found (404 status).",
                "url": url,
                "suggestion": "The page may have been moved or removed. Check the URL or try searching for the information elsewhere."
            }
            
        else:
            logger.warning(f"URL fetch failed with status code {response.status_code}")
            return {
                "error": f"Failed to fetch URL with status code {response.status_code}",
                "url": url,
                "suggestion": "Try a different URL or search for alternative sources of this information."
            }
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching URL {url}")
        return {
            "error": "The request timed out. The website might be slow or unavailable.",
            "url": url,
            "suggestion": "Try again later or use a different source."
        }
        
    except requests.exceptions.SSLError:
        logger.error(f"SSL Error while fetching URL {url}")
        return {
            "error": "SSL certificate verification failed. The website might have security issues.",
            "url": url,
            "suggestion": "Try a different source for this information."
        }
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection Error while fetching URL {url}")
        return {
            "error": "Failed to establish a connection to the server.",
            "url": url,
            "suggestion": "Check if the URL is correct or try a different source."
        }
        
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return {
            "error": str(e),
            "url": url,
            "message": "Failed to fetch content from this URL.",
            "suggestion": "Try a different URL or search for alternative sources."
        }

# Export the key classes and functions
__all__ = [
    'AzureAgent',
    'AgentBuilder',
    'run',
    'run_sync', 
    'run_async',
    'tool',
    'define_tool',
    'azure_function_tool',
    'azure_handoff',
    'create_tool_agent',
    'create_creative_agent',
    'create_agent',
    'connect_to_azure',
    'agent_context',
    'datetime_tool',
    'get_datetime_tool',
    'web_search_tool',
    'set_verbosity',
    'Tool',
    'FunctionTool',
    'ComputerTool',
    'WebSearchTool',
    'FileSearchTool',
    'InputGuardrail',
    'OutputGuardrail',
    'input_guardrail',
    'output_guardrail',
    'ModelSettings',
    'AgentHooks',
    'RunHooks',
    'ToolDefinition',
    'BaseContext'
]

# New simple function to create an agent with all capabilities in one go
def create_agent(
    name: str, 
    instructions: str,
    tools: Optional[List[Union[Callable, Tool, ToolDefinition]]] = None,
    handoffs: Optional[List[Union[Agent[Any], Handoff[Any]]]] = None,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
    use_search: bool = False,  # Simplify to use_search
    search_location: Optional[str] = None,  # Simplify to search_location
    use_chain_of_thought: bool = False,  # Simplify to use_chain_of_thought
    context_type: Optional[Type[Any]] = None,
    **kwargs
) -> AzureAgent:
    """
    Create an agent with a simple function call instead of using the builder pattern.
    
    Example:
    ```python
    # Create a simple assistant with web search
    agent = create_agent(
        name="My Assistant",
        instructions="You are a helpful assistant",
        tools=[calculate],
        temperature=0.7,
        use_search=True
    )
    ```
    
    Args:
        name: The name of the agent
        instructions: The instructions/system prompt for the agent
        tools: List of tools (functions or Tool instances). The datetime_tool is automatically included.
        handoffs: List of agents or handoffs this agent can delegate to
        temperature: Temperature setting (0-1)
        model_name: Azure OpenAI deployment name
        use_search: Whether to add web search capability
        search_location: Optional location for geographically relevant search results
        use_chain_of_thought: Whether to enhance with chain-of-thought reasoning
        context_type: Type hint for context
        **kwargs: Additional arguments for AzureAgent
        
    Returns:
        An AzureAgent configured with the specified options
    """
    # Initialize tools list if not provided
    if tools is None:
        tools = []
    
    # Convert to list in case it's a tuple
    tools_list = list(tools)
    
    # Add datetime_tool automatically to all agents
    # Check if datetime_tool is already in the list (by name)
    date_time_included = any(
        (isinstance(t, Tool) and getattr(t, "name", "") == "get_current_time") or
        (hasattr(t, "__name__") and t.__name__ == "datetime_tool")
        for t in tools_list
    )
    
    if not date_time_included:
        tools_list.append(datetime_tool)
    
    # Prepare instruction modifications if chain of thought is enabled
    final_instructions = instructions
    
    # Add built-in handoffs and tools guidance to all agents
    handoff_guidance = ""
    
    # Only add handoff guidance if there are handoffs
    if handoffs:
        # Create a list of available handoff descriptions for the prompt
        handoff_descriptions = []
        for h in handoffs:
            if isinstance(h, Handoff):
                tool_name = getattr(h, "tool_name", "specialized agent")
                tool_desc = getattr(h, "tool_description", "specialized tasks")
                handoff_descriptions.append(f"- {tool_name}: {tool_desc}")
            elif isinstance(h, Agent):
                h_name = h.name
                h_desc = getattr(h, "handoff_description", f"specialized in {h_name.lower()} tasks")
                handoff_descriptions.append(f"- {h_name}: {h_desc}")
        
        # Create the handoff guidance text
        if handoff_descriptions:
            handoff_guidance = f"""
HANDOFF GUIDANCE:
You have access to specialized agents who are experts in specific domains. ALWAYS hand off to these specialists when a query falls within their expertise:

{chr(10).join(handoff_descriptions)}

When deciding whether to use a handoff:
1. FIRST evaluate if the query involves specialized knowledge within any of these domains
2. If it does, you MUST use the appropriate handoff EVEN IF you think you can answer it
3. Do not attempt to solve specialized problems yourself when a more qualified agent is available
4. The purpose of handoffs is to provide users with the highest quality expertise

"""

    # Add tool usage guidance
    tools_guidance = ""
    if tools_list:
        tools_guidance = """
TOOL USAGE GUIDANCE:
You have access to various tools to help solve problems. Always:
1. Identify when a tool would be useful for answering a query
2. Use the most appropriate tool for the specific task
3. Interpret tool results accurately and include them in your response
4. Don't hesitate to use multiple tools if necessary to provide a complete answer
"""

    # Combine all guidance into the final instructions
    if handoff_guidance or tools_guidance:
        final_instructions = f"{instructions}\n\n{handoff_guidance}{tools_guidance}".strip()
    
    # Add chain of thought guidance if enabled
    if use_chain_of_thought:
        final_instructions = f"""
{final_instructions}

When solving problems or responding to complex queries, please think through your reasoning step by step:

1. First, understand what is being asked and identify key information
2. Break down complex problems into smaller parts
3. Think through each part methodically
4. Explain your reasoning process
5. Arrive at a well-reasoned conclusion

This step-by-step approach will help ensure your responses are thorough and accurate.
"""
        final_instructions = final_instructions.strip()
    
    # If handoffs exist, modify the target agent instructions to announce handoff
    if handoffs:
        for h in handoffs:
            if isinstance(h, Handoff):
                # Get the target agent - this happens via async, so we can't modify it here
                tool_desc = getattr(h, "tool_description", None)
                if tool_desc:
                    h.agent_name = name  # Store parent agent name
            elif isinstance(h, Agent):
                # Direct agent reference, we can modify its instructions
                agent_instructions = h.instructions
                if "At the beginning of your response, please indicate you're" not in agent_instructions:
                    h_name = h.name
                    h.instructions = f"{agent_instructions}\n\nAt the beginning of your response, include: \"I am the {h_name} and I'll help with your request.\" This is to make it clear to the user that a handoff has occurred."
    
    # Create the agent directly with all options
    return AzureAgent(
        name=name, 
        instructions=final_instructions,
        model_name=model_name,
        temperature=temperature,
        tools=tools_list,
        handoffs=handoffs,
        context_type=context_type,
        use_search=use_search,
        search_location=search_location,
        **kwargs
    ) 