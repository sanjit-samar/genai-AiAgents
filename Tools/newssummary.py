from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_results=5)

llm = ChatMistralAI(model="mistral-small-2603")

prompt = ChatPromptTemplate.from_template("""
you are a helpful news assistant
summarize the following news in meaningful context                                         
{news}
""")

parser = StrOutputParser()

chain = prompt | llm | parser

news_result = search_tool.run("what is the status of war 2026 between Iran and Usa")

result = chain.invoke({"news": news_result})

print(result)
