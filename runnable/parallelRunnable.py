from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

# 1. Load environment variables
load_dotenv()

# 2. Prompt Template
short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 2 lines")

detailed_prompt = ChatPromptTemplate.from_template("""
    You are an AI expert.

    Explain the following topic in simple terms:
    
    Topic: {topic}
    
    Give:
    - Short introduction
    - Key concepts
    - Real-world applications
    """)

# 3. Initialize Mistral Model
model = ChatMistralAI(model="mistral-small-2603", temperature=0.7)

# 4. Output Parser
parser = StrOutputParser()
