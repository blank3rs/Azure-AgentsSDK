# Azure Agents Framework

A powerful framework for building intelligent agents powered by Azure OpenAI, featuring seamless handoffs between specialized agents, web search capabilities, and flexible tool integration.

## Features

- 🤖 **Intelligent Agents**: Create specialized AI agents with different capabilities and personalities
- 🔄 **Agent Handoffs**: Seamless delegation between agents for specialized tasks
- 🔍 **Web Search Integration**: Built-in web search capabilities for real-time information
- 🛠️ **Flexible Tool System**: Easy integration of custom tools and functions
- 📝 **Chain of Thought**: Support for step-by-step reasoning
- 🌡️ **Temperature Control**: Fine-tune agent creativity and randomness
- 📊 **Context Management**: Track conversation history and agent handoffs
- 🔒 **Environment Management**: Secure configuration using environment variables

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` with your Azure OpenAI credentials.

## Configuration

Configure your Azure OpenAI credentials in the `.env` file:

```env
AZURE_OPENAI_KEY=your_azure_openai_key_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
```

## Usage

### Basic Agent Creation

```python
from azure_agents import create_agent, run

# Create a simple agent
agent = create_agent(
    name="My Assistant",
    instructions="You are a helpful assistant",
    temperature=0.7,
    use_search=True
)

# Run the agent
result = run(agent, "What can you help me with?")
print(result.final_output)
```

### Creating Specialized Agents with Handoffs

```python
# Create a specialized data analysis agent
data_agent = create_agent(
    name="Data Analysis Specialist",
    instructions="You are specialized in data analysis...",
    tools=[calculate],
    temperature=0.5
)

# Create main agent with handoff capability
main_agent = create_agent(
    name="Azure Assistant",
    instructions="You are a helpful assistant...",
    handoffs=[azure_handoff(data_agent, 
              tool_name="data_analysis_specialist",
              tool_description="Expert in data analysis")],
    use_search=True
)
```

### Adding Custom Tools

```python
from azure_agents import tool

@tool(name="calculate", description="Perform calculations")
def calculate(a: float, b: float, operation: str = "add") -> dict:
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Cannot divide by zero"
    }
    
    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}
        
    result = operations[operation](a, b)
    return {"result": result}
```

### Using Context Management

```python
from azure_agents import agent_context

class Context:
    def __init__(self):
        self.history = []
        self.handoffs = []
    
    def add_to_history(self, query, response):
        self.history.append((query, response))

# Run with context
with agent_context(Context()) as ctx:
    result = run(agent, "Your query here", ctx)
```

## Features in Detail

### Agent Handoffs
The framework supports intelligent handoffs between agents. When a specialized task is detected, the main agent can automatically delegate to a more specialized agent:

```python
# Handoff will be triggered automatically for data analysis queries
result = run(main_agent, "Can you analyze this dataset?")
```

### Web Search Integration
Agents can be created with web search capabilities enabled:

```python
agent = create_agent(
    name="Search Assistant",
    instructions="You can search the web...",
    use_search=True,
    search_location="New York, USA"  # Optional location context
)
```

### Chain of Thought Reasoning
Enable step-by-step reasoning:

```python
agent = create_agent(
    name="Reasoning Assistant",
    instructions="You help with complex problems...",
    use_chain_of_thought=True
)
```

## Best Practices

1. **Security**: Never commit your `.env` file. Always use environment variables for sensitive data.
2. **Temperature**: Use lower temperatures (0.1-0.3) for factual tasks and higher (0.7-0.9) for creative tasks.
3. **Context**: Implement proper context management for maintaining conversation history.
4. **Error Handling**: Always handle potential errors in custom tools.
5. **Documentation**: Document custom tools and agents thoroughly.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here] 