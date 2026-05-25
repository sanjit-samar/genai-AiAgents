from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 1. Load environment variables
load_dotenv()

# 3. Initialize Mistral Model
model = ChatMistralAI(model="mistral-small-2603", temperature=0.7)

# 4. Output Parser
parser = StrOutputParser()

# 2. Prompt Template
code_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a code generation"),
        ("human", "Hi, my name is {topic}"),
    ]
)

explain_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant who explain explains code in simple"),
        ("human", "Explain the following code in simple words\n{code}"),
    ]
)

sequence = code_prompt | model | parser

sequence2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explaination": explain_prompt | model | parser,
    }
)

# chian the response

chain = sequence | sequence2

response = chain.invoke({"topic": "write a square of two integers program"})

print(response["code"])
print(response["explaination"])
# print(response)
