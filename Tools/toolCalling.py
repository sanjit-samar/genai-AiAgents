from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from dotenv import load_dotenv
from rich import print

load_dotenv()

llm = ChatMistralAI(model="mistral-small-2603")


@tool
def get_text_length(text: str) -> int:
    """Return length of string"""
    return len(text)


# tool Bindding
llm_with_tools = llm.bind_tools([get_text_length])

# response = llm_with_tools.invoke("What is the length of the text : Hello")
# tool calling
response = get_text_length.invoke("How are you")
print(response)
