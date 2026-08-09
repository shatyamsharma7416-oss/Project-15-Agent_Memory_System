import chromadb
from chromadb.utils import embedding_functions

em_fun = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
DBclient = chromadb.PersistentClient("./my_chroma_data")
collection = DBclient.get_or_create_collection(
    name="agent_memory",
    embedding_function=em_fun
)

def retrieve_facts(query: str, top_k: int = 5) -> list[str]:
    """Semantic search — returns top_k relevant facts."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    if results and results["documents"]:
        return results["documents"][0]  # flatten list
    return []



retrieve_facts_schema = {
    "type": "function",
    "function":{
        "name": "retrieve_facts",
        "description": "Semantic search — returns top_k relevant facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query for retrieval from vector DB about user's facts",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar vector you want to retrieve. Default value is 5.",
                }
            },
            "required": ["query"],
        }
    }
}
