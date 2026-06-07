from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from rich import print

load_dotenv()


# custom Weather Tool
@tool
def weather_tool(city: str) -> str:
    """
    Get weather information for a city.
    """

    weather_data = {
        "bangalore": "28°C, Sunny",
        "mumbai": "31°C, Humid",
        "delhi": "35°C, Hot",
    }

    return weather_data.get(city.lower(), f"No weather data found for {city}")


# custom Calculator tool
@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate mathematical expressions.
    """

    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


# Search Tool
search_tool = TavilySearchResults(max_results=5)


# Registering all tools
tools = [search_tool, weather_tool, calculator_tool]

# memory by langgraph
memory = InMemorySaver()


# Create Agent
llm = ChatMistralAI(model="mistral-small-2603")

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a helpful AI assistant.

    Use tools whenever necessary.
    """,
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "user_123"}}

# Ask Questions
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "tell me what question I asked previously?",
                # "content": "What is the weather in Bangalore?",
                #  "content": "Calculate 25 * 17 + 100"
                #  "content": "Latest AI news today"
            }
        ]
    },
    config=config,
)

print(response["messages"][-1].content)
