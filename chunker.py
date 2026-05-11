import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


@dataclass
class Chunk:
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict
    strategy: str
    doc_id: int


class ImprovedUltraFastHybridChunker:
    def __init__(self, embedding_model: SentenceTransformer, similarity_threshold: float = 0.65,
                 min_chunk_size: int = 400, max_chunk_size: int = 1200, overlap_sentences: int = 2):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def estimate_entropy_fast(self, text: str) -> float:
        words = text.split()
        if len(words) == 0:
            return 0.5
        unique_ratio = len(set(words)) / len(words)
        avg_word_length = np.mean([len(w) for w in words])
        has_numbers = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        has_caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        has_math_symbols = sum(1 for c in text if c in '=+-*/^()[]{}') / max(len(text), 1)
        has_greek = sum(1 for c in text if c in 'αβγδεθλμσ∆') / max(len(text), 1)

        entropy_score = (unique_ratio * 0.35 + min(avg_word_length / 10, 1.0) * 0.25 +
                        min(has_numbers * 10, 1.0) * 0.15 + min(has_caps * 5, 1.0) * 0.10 +
                        min(has_math_symbols * 20, 1.0) * 0.10 + min(has_greek * 50, 1.0) * 0.05)
        return float(np.clip(entropy_score, 0, 1))

    def is_structural_boundary(self, sentence: str) -> bool:
        sentence_lower = sentence.lower().strip()
        if re.match(r'^\d+\.?\s+[A-Z]', sentence):
            return True
        if re.match(r'^(Theorem|Lemma|Corollary|Proposition|Definition|Algorithm|Figure)\s+\d+', sentence):
            return True
        if sentence_lower.startswith(('proof:', 'proof.', 'proof of theorem')):
            return True
        return False

    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        merged = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            if (i + 1 < len(sentences) and len(current) < 50 and
                not self.is_structural_boundary(current) and
                not self.is_structural_boundary(sentences[i + 1])):
                merged.append(current + ' ' + sentences[i + 1])
                i += 2
            else:
                merged.append(current)
                i += 1
        return merged

    def chunk(self, text: str, metadata: Dict = None, doc_id: int = 0) -> List[Chunk]:
        if metadata is None:
            metadata = {}
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return []
        sentences = self.preprocess_sentences(sentences)
        embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
        entropies = [self.estimate_entropy_fast(s) for s in sentences]

        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        chunk_idx = 0

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = len(sentence)
            sentence_entropy = entropies[i]
            is_boundary = self.is_structural_boundary(sentence)

            current_chunk_text = ' '.join(current_chunk)
            current_embedding = self.embedding_model.encode([current_chunk_text], show_progress_bar=False)[0]
            sentence_embedding = embeddings[i]

            similarity = cosine_similarity(current_embedding.reshape(1, -1),
                                         sentence_embedding.reshape(1, -1))[0][0]
            adjusted_threshold = self.similarity_threshold + (sentence_entropy * 0.10)

            if sentence_entropy > 0.7:
                adaptive_max = self.max_chunk_size * 0.7
            elif sentence_entropy > 0.5:
                adaptive_max = self.max_chunk_size * 0.85
            else:
                adaptive_max = self.max_chunk_size

            should_split = ((is_boundary and current_length >= self.min_chunk_size) or
                          (similarity < adjusted_threshold and current_length >= self.min_chunk_size) or
                          (current_length + sentence_length > adaptive_max))

            if should_split and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(text=chunk_text, start_idx=0, end_idx=len(chunk_text),
                    metadata={**metadata, 'chunk_idx': chunk_idx, 'sentence_count': len(current_chunk),
                             'chunk_length': len(chunk_text),
                             'avg_entropy': np.mean([entropies[j] for j in range(max(0, i - len(current_chunk)), i)])},
                    strategy='entropy_based', doc_id=doc_id))

                if self.overlap_sentences > 0 and len(current_chunk) > self.overlap_sentences:
                    current_chunk = current_chunk[-self.overlap_sentences:]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
                chunk_idx += 1

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(text=chunk_text, start_idx=0, end_idx=len(chunk_text),
                metadata={**metadata, 'chunk_idx': chunk_idx, 'sentence_count': len(current_chunk),
                         'chunk_length': len(chunk_text),
                         'avg_entropy': np.mean(entropies[-len(current_chunk):]) if current_chunk else 0},
                strategy='entropy_based', doc_id=doc_id))
        return chunks