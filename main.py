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
    azure_handoff,
    BaseContext
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


coder_agent = create_agent(
    name="Coder",
    instructions="""You are a skilled programmer and coding expert. Your expertise includes:

1. Writing efficient algorithms
2. Debugging and troubleshooting code
3. Implementing data structures and algorithms
4. Explaining complex coding concepts clearly
5. Providing insights into software design patterns and best practices

When users have questions related to coding, algorithms, or software development that are complex, you'll handle these tasks efficiently and provide expert guidance.

IMPORTANT: You must ALWAYS maintain context of the entire conversation history. If a user refers to something mentioned earlier, you must remember it and respond appropriately.""",
    temperature=0.5,
    use_chain_of_thought=True
)

data_analysis_agent = create_agent(
    name="Data Analysis Specialist",
    instructions="""You are a specialized data analysis assistant. Your expertise includes:

1. Statistical analysis and interpretation
2. Data visualization techniques
3. Working with datasets and numerical information
4. Explaining complex data concepts clearly
5. Providing insights into numerical trends and patterns

When users have questions related to data analysis, statistics, or numerical calculations that are complex, you'll handle these tasks efficiently and provide expert guidance.

IMPORTANT: You must ALWAYS maintain context of the entire conversation history. If a user refers to something mentioned earlier, you must remember it and respond appropriately.""",
    tools=[calculate],
    temperature=0.5,
    use_chain_of_thought=True,
    handoffs=[azure_handoff(coder_agent,tool_name="Coder",tool_description="EXPERT in COding and can code anything")]
)

# Create an agent with web search capability and handoff to specialized agent
agent = create_agent(
    name="Azure Assistant",
    instructions="""You are a helpful conversational assistant with web search capabilities and PERFECT MEMORY of the entire conversation.

CONVERSATION CONTEXT HANDLING (CRITICAL):
- You MUST treat this as a continuous conversation where EVERY previous message provides context
- You MUST remember and refer back to information from previous messages
- When users make short requests or use pronouns (it, they, that, etc.), you MUST understand they are referring to topics from earlier in the conversation
- NEVER ask for clarification about things already mentioned in the conversation
- ALWAYS track and recall names, facts, and details from earlier in the conversation
- If a user says "check again" or similar phrases, they are asking you to revisit or reconsider a previous topic

When users ask about current information, follow these steps:
1. Use the web_search tool to find relevant information
2. Review the search results
3. If needed, use the fetch_url tool to retrieve more detailed information from specific URLs
4. You can continue fetching additional URLs from the page links if necessary
5. Once you have enough information, provide a comprehensive answer

When presenting information:
- Cite your sources when appropriate
- Include relevant details but keep your response concise
- Maintain a helpful and informative tone
- Present the information naturally as part of your response

For search queries only respond with what I asked for, don't say "for more info go here" or similar phrases.

Your primary goal is to maintain PERFECT CONTINUITY in the conversation by remembering everything previously discussed.""",
    tools=[calculate],  
    handoffs=[azure_handoff(data_analysis_agent, 
                          tool_name="data_analysis_specialist", 
                          tool_description="Expert in statistical analysis, data interpretation, and complex calculations")],
    temperature=0.7,
    use_search=True,
    use_chain_of_thought=True
)

# Run the agent
if __name__ == "__main__":
    # Create a context for the conversation with a descriptive name
    context = BaseContext()
    
    print(f"\n{'='*80}")
    print("Azure Agent Interactive Chat")
    print("Type 'exit', 'quit', or 'q' to end the conversation")
    print("Type 'context' to see the current conversation context")
    print(f"{'='*80}")
    
    query_count = 0
    while True:
        query = input("\nEnter your query: ")
        
        # Check for exit commands
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        # Special command to display context
        if query.lower() == "context":
            print("\n=== CURRENT CONVERSATION CONTEXT ===")
            if context.history:
                for i, entry in enumerate(context.history):
                    print(f"{i+1}. User: {entry.get('query', '')[:50]}...")
                    print(f"   Assistant: {entry.get('response', '')[:50]}...")
                    print("-" * 40)
            else:
                print("No conversation history yet.")
            continue
            
        if not query.strip():
            print("Please enter a valid query.")
            continue
            
        query_count += 1
        print(f"\n{'='*80}")
        print(f"QUERY {query_count}: {query}")
        print(f"{'='*80}")
        
        try:
            # Run the agent with the context
            result = run(agent, query, context)
            print(f"\n{result.final_output}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Print handoff summary at the end
    print("\n" + "="*80)
    print("CONVERSATION SUMMARY")
    print("="*80)
    
    # Show the number of exchanges
    print(f"\nTotal exchanges: {len(context.get_history())}")
    
    # Show handoffs using BaseContext's tracking
    handoffs = context.get_handoffs()
    if handoffs:
        print("\nHANDOFF SUMMARY:")
        print("-"*40)
        for i, handoff in enumerate(handoffs):
            print(f"{i+1}. From '{handoff['from_agent']}' to '{handoff['to_agent']}'")
    else:
        print("\nNo handoffs occurred during this session.")

