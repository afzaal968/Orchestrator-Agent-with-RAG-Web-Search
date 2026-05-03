from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tools import tavily_search
import json
import os
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global storage for book chunks and embeddings
book_embeddings = None
book_chunks = None

def get_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def get_embeddings(texts):
    """Get embeddings from OpenAI for a list of texts"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_book(query, top_k=3):
    """Search for relevant chunks in the book using embeddings"""
    global book_embeddings, book_chunks
    
    if book_embeddings is None or book_chunks is None:
        return []
    
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # Calculate similarities
    similarities = [
        (i, cosine_similarity(query_embedding, chunk_emb))
        for i, chunk_emb in enumerate(book_embeddings)
    ]
    
    # Sort by similarity and get top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [book_chunks[i] for i, _ in similarities[:top_k]]
    
    return top_chunks

def initialize_book_knowledge():
    """Load and process the Atomic Habits book"""
    global book_embeddings, book_chunks
    
    print("Loading Atomic Habits book...")
    pdf_path = "Atomic habits ( PDFDrive ).pdf"
    
    # Extract text
    text = get_pdf_text(pdf_path)
    
    # Create chunks
    print("Creating text chunks...")
    book_chunks = chunk_text(text)
    
    # Get embeddings (in batches to avoid rate limits)
    print(f"Creating embeddings for {len(book_chunks)} chunks...")
    book_embeddings = get_embeddings(book_chunks)
    
    print("Book knowledge base initialized!\n")

def classify_intent(user_query):
    """Classify user intent using OpenAI"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a query classifier. Classify the user's query into one of these categories:
1. "atomic_habits" - Questions about the Atomic Habits book, habits, behavior change, James Clear
2. "search" - Questions about current events, news, latest information, web search queries
3. "other" - Any other questions

Respond with ONLY the category name: atomic_habits, search, or other"""
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip().lower()

def answer_from_book(query):
    """Answer question using RAG from Atomic Habits book"""
    # Get relevant chunks
    relevant_chunks = search_book(query)
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the book."
    
    # Create context from chunks
    context = "\n\n".join(relevant_chunks)
    
    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions about the Atomic Habits book. Use only the provided context to answer questions. If the context doesn't contain the answer, say so."
            },
            {
                "role": "user",
                "content": f"Context from Atomic Habits:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def answer_with_search(query):
    """Answer question using Tavily web search"""
    print("Searching the web...")
    search_results = tavily_search(query=query, max_results=5)
    
    # Extract relevant information
    context = ""
    for result in search_results.get('results', []):
        context += f"Source: {result['title']}\n{result['content']}\n\n"
    
    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided search results to answer the user's question accurately."
            },
            {
                "role": "user",
                "content": f"Search Results:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def orchestrator(user_query):
    """Main orchestrator that routes queries to the appropriate handler"""
    print(f"\nUser: {user_query}")
    print("Classifying intent...")
    
    intent = classify_intent(user_query)
    print(f"Intent: {intent}\n")
    
    if intent == "atomic_habits":
        print("Using RAG search on Atomic Habits book...")
        answer = answer_from_book(user_query)
    elif intent == "search":
        answer = answer_with_search(user_query)
    else:
        answer = "I'm sorry, I can only help with questions about the Atomic Habits book or search for current information on the web."
    
    print(f"\nAssistant: {answer}\n")
    print("-" * 80)
    return answer

def main():
    load_dotenv()
    
    # Initialize the book knowledge base
    initialize_book_knowledge()
    
    print("=" * 80)
    print("ORCHESTRATOR AGENT - Atomic Habits RAG + Web Search")
    print("=" * 80)
    print("I can help you with:")
    print("1. Questions about the Atomic Habits book")
    print("2. Search for latest news and information")
    print("\nType 'quit' to exit\n")
    print("=" * 80)
    
    # Interactive loop
    while True:
        user_query = input("\nYou: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_query:
            continue
        
        orchestrator(user_query)

if __name__ == '__main__':
    main()