from dotenv import load_dotenv
from rich import print
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# LLM
llm = ChatMistralAI(model="mistral-small-2603")

# Tool
search_tool = TavilySearchResults(max_results=5)

tools = [search_tool]

# Prompt
prompt = """
You are a research assistant.

When solving a problem:

1. Analyze the question carefully.
2. If external information is needed, use available tools.
3. Evaluate tool results before answering.
4. Give accurate and concise answers.
5. Cite evidence from tool outputs when possible.
"""

# Create Agent
agent = create_agent(model=llm, tools=tools, system_prompt=prompt)

# Run Agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Latest Iran USA war update?"}]}
)

print(response)
