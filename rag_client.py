import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("chroma_db")]

    # TODO: Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # TODO: Wrap connection attempt in try-except block for error handling
        try:
            # TODO: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )

            # TODO: Retrieve list of available collections from the database
            collections = client.list_collections()

            # TODO: Loop through each collection found
            for collection in collections:
                # TODO: Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}/{collection.name}"
                # TODO: Build information dictionary containing:
                # TODO: Store directory path as string
                # TODO: Store collection name
                # TODO: Create user-friendly display name
                count = client.get_collection(name=collection.name).count()
                # TODO: Add collection information to backends dictionary
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} - {collection.name} ({count} docs)",
                    "count": count
                }

        # TODO: Handle connection or access errors gracefully
        except Exception as e:
            # TODO: Create fallback entry for inaccessible directories
            # TODO: Include error information in display name with truncation
            # TODO: Set appropriate fallback values for missing information
            backends[chroma_dir.name] = {
                "directory": str(chroma_dir),
                "collection_name": None,
                "display_name": f"{chroma_dir.name} - Error: {str(e)[:50]}",
                "count": 0
            }

    # TODO: Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # TODO: Create a chomadb persistentclient
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    # TODO: Return the collection with the collection_name
    return client.get_collection(name=collection_name), True, None

def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    where = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
    if mission_filter and mission_filter.lower() != "all":
        where = {"mission": mission_filter}

    # TODO: Execute database query with the following parameters:
        # TODO: Pass search query in the required format
        # TODO: Set maximum number of results to return
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    import os
    openai_key = os.environ.get("CHROMA_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        client = OpenAI(api_key=openai_key, base_url="https://openai.vocareum.com/v1")
        embedding = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Deduplicate by document content while preserving order
    seen = set()
    unique_docs = []
    unique_metas = []
    for doc, meta in zip(documents, metadatas):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_metas.append(meta)

    # TODO: Initialize list with header text for context section
    context_parts = ["### Retrieved Context ###\n"]

    # TODO: Loop through paired documents and their metadata using enumeration
    for i, (doc, metadata) in enumerate(zip(unique_docs, unique_metas), 1):
        # TODO: Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown")
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()
        # TODO: Extract source information from metadata with fallback value
        source = metadata.get("source", "unknown")
        # TODO: Extract category information from metadata with fallback value
        category = metadata.get("document_category", "unknown")
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()

        # TODO: Create formatted source header with index number and extracted information
        # TODO: Add source header to context parts list
        context_parts.append(f"[Source {i}] Mission: {mission} | File: {source} | Category: {category}")

        # TODO: Check document length and truncate if necessary
        # TODO: Add truncated or full document content to context parts list
        if len(doc) > 500:
            context_parts.append(doc[:500] + "...")
        else:
            context_parts.append(doc)

    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
