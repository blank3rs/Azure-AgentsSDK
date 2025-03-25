"""
Azure Agents with flexible web search capability and handoff notification
"""

from azure_agents import (
    create_agent,
    run, 
    tool, 
    agent_context,
    set_verbosity,
    logging,
    azure_handoff
)

# Enable logging
set_verbosity(logging.INFO)

# Define a calculation tool
@tool(name="calculate", description="Perform a calculation on two numbers")
def calculate(a: float, b: float, operation: str = "add") -> dict:
    """
    Calculate the result of a mathematical operation.
    
    Args:
        a: First number
        b: Second number
        operation: One of: add, subtract, multiply, divide
    """
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

# Create a specialized agent for data analysis tasks
data_analysis_agent = create_agent(
    name="Data Analysis Specialist",
    instructions="""You are a specialized data analysis assistant. Your expertise includes:

1. Statistical analysis and interpretation
2. Data visualization techniques
3. Working with datasets and numerical information
4. Explaining complex data concepts clearly
5. Providing insights into numerical trends and patterns

When users have questions related to data analysis, statistics, or numerical calculations that are complex, you'll handle these tasks efficiently and provide expert guidance.""",
    tools=[calculate],
    temperature=0.5,
    use_chain_of_thought=True
)

# Create an agent with web search capability and handoff to specialized agent
agent = create_agent(
    name="Azure Assistant",
    instructions="""You are a helpful assistant with web search capabilities. When users ask about current information, follow these steps:

1. Use the web_search tool to find relevant information
2. Review the search results
3. If needed, use the fetch_url tool to retrieve more detailed information from specific URLs
4. You can continue fetching additional URLs from the page links if necessary
5. Once you have enough information, provide a comprehensive answer

For complex data analysis or statistical questions, consider handing off to the Data Analysis Specialist.

When presenting information:
- Cite your sources when appropriate
- Include relevant details but keep your response concise
- Maintain a helpful and informative tone
- Present the information naturally as part of your response

For search queries only respond with what I asked for, don't say "for more info go here" or similar phrases.
Your goal is to be as helpful as possible by providing relevant, accurate information.""",
    tools=[calculate],  
    handoffs=[azure_handoff(data_analysis_agent, 
                          tool_name="data_analysis_specialist", 
                          tool_description="Expert in statistical analysis, data interpretation, and complex calculations")],
    temperature=0.7,
    use_search=True,
    use_chain_of_thought=True
)

# Enhanced Context for conversation memory that tracks handoffs
class Context:
    def __init__(self):
        self.history = []
        self.handoffs = []
    
    def add_to_history(self, query, response):
        self.history.append((query, response))
        
    def get_history(self):
        return self.history
    
    def add_handoff(self, from_agent, to_agent):
        """Track handoffs between agents"""
        self.handoffs.append((from_agent, to_agent))
        
    def get_handoffs(self):
        return self.handoffs

# Run the agent
if __name__ == "__main__":
    # Test queries covering different types of information
    queries = [
        "Calculate the compound interest on $1000 invested for 5 years at an annual rate of 8%",
        "What is the standard deviation and why is it important in data analysis?",
        "What's the current weather in New York?",
        "Can you perform a t-test analysis on this dataset: [65, 78, 88, 55, 48, 95, 66, 57, 79, 81]?"
    ]
    
    # Run the agent with a context
    with agent_context(Context()) as ctx:
        for i, query in enumerate(queries):
            print(f"\n{'='*80}")
            print(f"QUERY {i+1}: {query}")
            print(f"{'='*80}")
            try:
                result = run(agent, query, ctx)
                print(f"\n{result.final_output}")
                
                # Add to conversation history
                ctx.add_to_history(query, result.final_output)
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Print handoff summary at the end
        print("\n" + "="*80)
        print("HANDOFF SUMMARY:")
        print("="*80)
        
        if ctx.handoffs:
            for i, (from_agent, to_agent) in enumerate(ctx.handoffs):
                print(f"{i+1}. From '{from_agent}' to '{to_agent}'")
        else:
            print("No handoffs occurred during this session.")

