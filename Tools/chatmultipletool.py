from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from rich import print
from rich.console import Console
from rich.prompt import Prompt

load_dotenv()

console = Console()


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

# ── Conversation Loop ──────────────────────────────────────────────────────────
console.print("\n[bold green]🤖 AI Assistant started![/bold green]")
console.print("[dim]Type your question or 'quit' / press Enter to exit.[/dim]\n")

while True:
    user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

    # Skip empty input — don't exit, just re-prompt
    if not user_input:
        console.print("[dim]Please enter a question or type 'quit' to exit.[/dim]")
        continue

    # Exit only on explicit "quit"
    if user_input.lower() == "quit":
        console.print("\n[bold yellow]👋 Ending conversation. Goodbye![/bold yellow]")
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        },
        config=config,
    )

    answer = response["messages"][-1].content
    console.print(f"\n[bold magenta]Assistant:[/bold magenta] {answer}\n")
