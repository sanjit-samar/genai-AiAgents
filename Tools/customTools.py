from langchain.tools import tool


@tool  # Decorator for tool
def get_greeting(name: str) -> str:
    # Doc string
    """Generate a greeting message for a user"""
    return f"Hello {name} Welcome back !"


result = get_greeting.invoke({"name": "sanjit"})
print(result)
