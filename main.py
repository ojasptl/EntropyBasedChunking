from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from chunker import ImprovedUltraFastHybridChunker
from pdf_utils import process_pdf
from retriever import BM25Index, hybrid_search, print_results
from llm import initialize_llm, generate_answer


def initialize_rag_system():
    print("🚀 Initializing RAG System...")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    llm_model, llm_tokenizer = initialize_llm()

    qdrant_client = QdrantClient(":memory:")
    chunker = ImprovedUltraFastHybridChunker(embedding_model=embedding_model)

    print("✅ System initialized!\n")
    return embedding_model, llm_model, llm_tokenizer, qdrant_client, chunker


def search_and_answer(query: str, collection_name: str,
                      embedding_model, qdrant_client, bm25_index: BM25Index,
                      llm_model, llm_tokenizer, top_k: int = 5):
    print(f"\n🔍 Searching for: '{query}'\n")

    results = hybrid_search(
        query=query,
        collection_name=collection_name,
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        bm25_index=bm25_index,
        top_k=top_k,
    )

    context = print_results(results, query)

    print("\n💡 Generating Answer...\n")
    answer = generate_answer(query, context, llm_model, llm_tokenizer)

    print("=" * 80)
    print(f"Q: {query}")
    print(f"\nA: {answer.strip()}")
    print("=" * 80)


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embedding_model, llm_model, llm_tokenizer, qdrant_client, chunker = initialize_rag_system()

    pdf_path = input("Enter PDF file path: ").strip()
    collection_name, chunk_texts = process_pdf(pdf_path, embedding_model, qdrant_client, chunker)

    if collection_name:
        bm25_index = BM25Index(chunk_texts)
        print("✅ BM25 index built\n")

        while True:
            print("\n" + "=" * 80)
            query = input("\nEnter your question (or 'quit' to exit): ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if query:
                top_k = input("Top K results (default 5): ").strip()
                top_k = int(top_k) if top_k.isdigit() else 5

                search_and_answer(query, collection_name, embedding_model,
                                  qdrant_client, bm25_index,
                                  llm_model, llm_tokenizer, top_k)
            else:
                print("⚠️  Please enter a question!")