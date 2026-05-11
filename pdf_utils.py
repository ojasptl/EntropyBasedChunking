import PyPDF2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from chunker import ImprovedUltraFastHybridChunker


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"❌ Error extracting PDF: {str(e)}")
        return ""


def process_pdf(pdf_path: str, embedding_model: SentenceTransformer,
                qdrant_client: QdrantClient, chunker: ImprovedUltraFastHybridChunker):
    print(f"\n📄 Processing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("❌ No text extracted")
        return None, []

    print(f"✅ Extracted {len(text)} characters")
    chunks = chunker.chunk(text)
    print(f"✅ Created {len(chunks)} chunks (avg size: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars)")

    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)

    collection_name = "pdf_collection"
    try:
        qdrant_client.get_collection(collection_name)
        qdrant_client.delete_collection(collection_name)
    except:
        pass

    qdrant_client.create_collection(collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE))

    points = [PointStruct(id=idx, vector=embeddings[idx].tolist(),
        payload={"text": chunks[idx].text, "chunk_idx": chunks[idx].metadata.get('chunk_idx', idx),
                "sentence_count": chunks[idx].metadata.get('sentence_count', 0),
                "chunk_length": chunks[idx].metadata.get('chunk_length', len(chunks[idx].text))})
        for idx in range(len(chunks))]

    qdrant_client.upsert(collection_name=collection_name, points=points)
    print("✅ Stored in vector database\n")
    # Return collection name AND raw chunk texts for BM25 corpus
    return collection_name, chunk_texts