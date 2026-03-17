from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str], reference: Optional[str] = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available. Install with: pip install ragas"}

    # Validate inputs - return clear error messages instead of crashing
    if not question or not question.strip():
        return {"error": "Question cannot be empty"}
    if not answer or not answer.strip():
        return {"error": "Answer cannot be empty"}
    if not contexts or not isinstance(contexts, list) or not any(c.strip() for c in contexts if c):
        return {"error": "Contexts must be a non-empty list of strings"}

    import asyncio

    # Ensure OPENAI_API_KEY is set for LangChain wrappers (chat.py stores it as CHROMA_OPENAI_API_KEY)
    if not os.environ.get("OPENAI_API_KEY"):
        fallback = os.environ.get("CHROMA_OPENAI_API_KEY")
        if fallback:
            os.environ["OPENAI_API_KEY"] = fallback

    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-3.5-turbo", base_url="https://openai.vocareum.com/v1")
    )
    # TODO: Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", base_url="https://openai.vocareum.com/v1")
    )

    # TODO: Define an instance for each metric to evaluate
    # ResponseRelevancy and Faithfulness run on every call.
    # BleuScore and RougeScore require a ground-truth reference and are included only when one is provided.
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm),
    ]
    if reference:
        metrics += [BleuScore(), RougeScore()]

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference
    )

    # TODO: Evaluate the response using the metrics
    scores = {}
    for metric in metrics:
        scores[metric.name] = asyncio.run(metric.single_turn_ascore(sample))

    # TODO: Return the evaluation results
    return scores


def load_evaluation_dataset(filepath: str) -> List[Dict[str, str]]:
    """
    Load evaluation questions from a dataset file.

    Expected format (one question per line):
        category | question text

    Lines starting with '#' or blank lines are skipped.

    Returns:
        List of dicts with 'category' and 'question' keys.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file contains no valid questions.
    """
    try:
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '|' in line:
                    parts = line.split('|', 2)
                    category = parts[0].strip()
                    question = parts[1].strip()
                    reference = parts[2].strip() if len(parts) > 2 else None
                else:
                    category = 'general'
                    question = line.strip()
                    reference = None
                if question:
                    questions.append({'category': category, 'question': question, 'reference': reference})
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation dataset not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading evaluation dataset '{filepath}': {e}")

    if not questions:
        raise ValueError(f"No valid questions found in '{filepath}'")

    return questions


def run_batch_evaluation(dataset_path: str, chroma_dir: str, collection_name: str,
                         openai_key: str, n_results: int = 3) -> Dict:
    """
    Run batch evaluation on the evaluation dataset end-to-end.

    For each question:
      1. Retrieves context from ChromaDB
      2. Generates an LLM answer
      3. Evaluates with RAGAS metrics

    Outputs a per-question summary and aggregate (mean) scores across all questions.

    Args:
        dataset_path:     Path to evaluation_dataset.txt
        chroma_dir:       ChromaDB persist directory
        collection_name:  ChromaDB collection name
        openai_key:       OpenAI API key
        n_results:        Number of documents to retrieve per question

    Returns:
        Dict with 'results' (per-question) and 'aggregates' (mean per metric).
    """
    import rag_client
    import llm_client

    os.environ["OPENAI_API_KEY"] = openai_key

    # Load questions from dataset file
    questions = load_evaluation_dataset(dataset_path)
    print(f"\nLoaded {len(questions)} questions from '{dataset_path}'\n")

    # Initialize RAG system
    try:
        collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
        if not success:
            raise RuntimeError(f"Failed to initialize RAG system: {error}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAG system: {e}")

    results = []
    metric_totals: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    for i, item in enumerate(questions, 1):
        question = item['question']
        category = item['category']
        print(f"[{i}/{len(questions)}] Category: {category}")
        print(f"  Question: {question}")

        # Retrieve context from ChromaDB
        docs_result = rag_client.retrieve_documents(collection, question, n_results=n_results)
        contexts = []
        context_str = ""
        if docs_result and docs_result.get("documents"):
            contexts = docs_result["documents"][0]
            context_str = rag_client.format_context(
                docs_result["documents"][0], docs_result["metadatas"][0]
            )

        # Generate LLM answer grounded in retrieved context
        answer = llm_client.generate_response(openai_key, question, context_str, [])

        # Evaluate with RAGAS metrics
        scores = evaluate_response_quality(question, answer, contexts, reference=item.get('reference'))

        result = {
            'question': question,
            'category': category,
            'answer': answer,
            'scores': scores,
        }
        results.append(result)

        # Print per-question summary
        print(f"  Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print(f"  Scores:")
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")
                metric_totals[metric] = metric_totals.get(metric, 0.0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
            else:
                print(f"    {metric}: {value}")
        print()

    # Compute and print aggregate (mean) scores
    aggregates = {m: metric_totals[m] / metric_counts[m] for m in metric_totals}
    print("=" * 60)
    print("AGGREGATE SCORES (mean across all questions)")
    print("=" * 60)
    for metric, value in aggregates.items():
        print(f"  {metric}: {value:.4f}")

    return {'results': results, 'aggregates': aggregates}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch RAGAS evaluation for the NASA Mission Intelligence RAG system'
    )
    parser.add_argument('--dataset', default='evaluation_dataset.txt',
                        help='Path to evaluation dataset (default: evaluation_dataset.txt)')
    parser.add_argument('--chroma-dir', required=True,
                        help='ChromaDB persist directory (e.g. ./chroma_db_openai)')
    parser.add_argument('--collection-name', required=True,
                        help='ChromaDB collection name (e.g. nasa_space_missions_text)')
    parser.add_argument('--openai-key', required=True,
                        help='OpenAI API key')
    parser.add_argument('--n-results', type=int, default=3,
                        help='Number of documents to retrieve per question (default: 3)')
    args = parser.parse_args()

    run_batch_evaluation(
        dataset_path=args.dataset,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        openai_key=args.openai_key,
        n_results=args.n_results,
    )
