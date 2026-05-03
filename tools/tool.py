from tavily import TavilyClient
import os 
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_search(query: str, max_results: int = 5):
    """Search the web using Tavily."""
    response = tavily.search(
        query=query,
        max_results=max_results
    )
    return response

