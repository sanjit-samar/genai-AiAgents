from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 1. Load environment variables
load_dotenv()

# 3. Initialize Mistral Model
model = ChatMistralAI(model="mistral-small-2603", temperature=0.7)

# 4. Output Parser
parser = StrOutputParser()

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

# For invoking runnables with same value

# chains = RunnableParallel(
#     {
#         "short": short_prompt | model | parser,
#         "detailed": detailed_prompt | model | parser,
#     }
# )

# result = chains.invoke({"topic": "generative Ai"})

# For invoking runnables with different different value with RunnableLambda

chains = RunnableParallel(
    {
        "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
        "detailed": RunnableLambda(lambda x: x["detailed"])
        | detailed_prompt
        | model
        | parser,
    }
)

result = chains.invoke(
    {"short": {"topic": "generative Ai"}, "detailed": {"topic": "Machine Learning"}}
)

print(result)
