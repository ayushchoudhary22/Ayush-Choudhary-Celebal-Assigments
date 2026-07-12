import os
import re
import json
import math
import numpy as np
from pypdf import PdfReader

# ============================================================
# STOP WORDS - filtered from embeddings for better matching
# ============================================================
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an',
    'and', 'any', 'are', 'aren', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could', 'd', 'did',
    'do', 'does', 'doing', 'don', 'down', 'during', 'each', 'even', 'few', 'for',
    'from', 'further', 'get', 'got', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'let', 'll', 'm', 'may',
    'me', 'might', 'more', 'most', 'must', 'my', 'myself', 'need', 'no', 'nor',
    'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shall', 'she',
    'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 'we', 'were',
    'weren', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
    'with', 'won', 'would', 'wouldn', 'y', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'etc', 'using', 'used', 'use',
}

EMBEDDING_DIMS = 2048


def tokenize(text: str) -> list[str]:
    """Tokenize into lowercase words, removing stop words."""
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def _hash_token(token: str, dims: int = EMBEDDING_DIMS) -> int:
    h = 0
    for c in token:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h % dims


def _hash_sign(token: str) -> float:
    h = 0
    for c in token:
        h = (h * 37 + ord(c)) & 0xFFFFFFFF
    return 1.0 if (h % 2) == 0 else -1.0


def get_fallback_embedding(text: str, dims: int = EMBEDDING_DIMS) -> list[float]:
    """TF-IDF hashing vectorizer: stop words + unigrams + bigrams + sublinear TF."""
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dims

    features = list(tokens)
    for i in range(len(tokens) - 1):
        features.append(tokens[i] + '_' + tokens[i + 1])

    tf = {}
    for f in features:
        tf[f] = tf.get(f, 0) + 1

    vec = np.zeros(dims, dtype=np.float64)
    for token, count in tf.items():
        idx = _hash_token(token, dims)
        sign = _hash_sign(token)
        weight = 1.0 + math.log(count) if count > 1 else 1.0
        vec[idx] += sign * weight

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ============================================================
# PDF Extraction with Deduplication
# ============================================================
def deduplicate_text(text: str) -> str:
    """Remove duplicate sentences and repeated phrases from PDF text."""
    # Step 1: Remove repeated phrases (sliding window)
    words = text.split()
    cleaned_words = []
    i = 0
    while i < len(words):
        found_repeat = False
        for phrase_len in range(3, min(16, (len(words) - i) // 2 + 1)):
            phrase = words[i:i + phrase_len]
            next_phrase = words[i + phrase_len:i + 2 * phrase_len]
            if phrase == next_phrase:
                cleaned_words.extend(phrase)
                i += 2 * phrase_len
                found_repeat = True
                break
        if not found_repeat:
            cleaned_words.append(words[i])
            i += 1
    text = ' '.join(cleaned_words)

    # Step 2: Remove duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for sent in sentences:
        normalized = sent.strip().lower()
        if len(normalized) < 10:
            unique.append(sent.strip())
            continue
        if normalized not in seen:
            seen.add(normalized)
            unique.append(sent.strip())
    return ' '.join(unique)


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    pages_data = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                cleaned = re.sub(r'\s+', ' ', text).strip()
                cleaned = deduplicate_text(cleaned)
                if cleaned and len(cleaned) > 20:
                    pages_data.append({"text": cleaned, "page_num": i + 1})
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return pages_data


def chunk_document(pdf_path: str, chunk_size: int = 600, chunk_overlap: int = 80) -> list[dict]:
    filename = os.path.basename(pdf_path)
    pages_data = extract_text_from_pdf(pdf_path)
    chunks = []

    for page in pages_data:
        text = page["text"]
        page_num = page["page_num"]
        start = 0
        chunk_idx = 0

        while start < len(text):
            if start + chunk_size >= len(text):
                chunk_text = text[start:].strip()
                if chunk_text and len(chunk_text) > 30:
                    chunks.append({
                        "id": f"{filename}_p{page_num}_c{chunk_idx}",
                        "text": chunk_text,
                        "source": filename,
                        "page": page_num,
                        "chunk_index": chunk_idx
                    })
                break

            end = start + chunk_size
            last_period = text.rfind('. ', start + int(chunk_size * 0.5), end)
            if last_period != -1:
                end = last_period + 1
            else:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > (start + int(chunk_size * 0.6)):
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text and len(chunk_text) > 30:
                chunks.append({
                    "id": f"{filename}_p{page_num}_c{chunk_idx}",
                    "text": chunk_text,
                    "source": filename,
                    "page": page_num,
                    "chunk_index": chunk_idx
                })
                chunk_idx += 1

            start = end - chunk_overlap
            if start < 0:
                start = 0

    return chunks


# ============================================================
# Vector Store with Cosine Similarity
# ============================================================
class SimpleVectorStore:
    def __init__(self, storage_path: str = "vector_store.json"):
        self.storage_path = storage_path
        self.chunks = []
        self.embeddings = []
        self.load()

    def add_documents(self, new_chunks: list[dict], new_embeddings: list[list[float]]):
        if len(new_chunks) != len(new_embeddings):
            raise ValueError("Lengths of chunks and embeddings must match.")
        for chunk, emb in zip(new_chunks, new_embeddings):
            existing_idx = next((i for i, c in enumerate(self.chunks) if c["id"] == chunk["id"]), None)
            if existing_idx is not None:
                self.chunks[existing_idx] = chunk
                self.embeddings[existing_idx] = emb
            else:
                self.chunks.append(chunk)
                self.embeddings.append(emb)
        self.save()

    def similarity_search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """
        Calculates cosine similarity between the query embedding and stored embeddings.
        Returns the top_k most similar chunks, including similarity scores.
        """
        if not self.embeddings:
            return []
        query_vec = np.array(query_embedding, dtype=np.float64)
        all_vecs = np.array(self.embeddings, dtype=np.float64)

        dot_product = np.dot(all_vecs, query_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_vectors = np.linalg.norm(all_vecs, axis=1)
        norm_product = norm_query * norm_vectors
        norm_product[norm_product == 0] = 1e-10

        similarities = dot_product / norm_product
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices[:top_k]:
            result_chunk = self.chunks[idx].copy()
            result_chunk["similarity_score"] = float(similarities[idx])
            result_chunk["hybrid_score"] = float(similarities[idx]) # fallback
            results.append(result_chunk)
        return results

    def hybrid_search(self, query: str, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """
        Combines BM25 lexical score and TF-IDF vector cosine similarity for robust retrieval.
        Gives high priority to rare search words like names (e.g. "Ayush", "Choudhary").
        """
        if not self.chunks:
            return []

        # 1. Cosine similarity
        query_vec = np.array(query_embedding, dtype=np.float64)
        all_vecs = np.array(self.embeddings, dtype=np.float64)

        dot_product = np.dot(all_vecs, query_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_vectors = np.linalg.norm(all_vecs, axis=1)
        norm_product = norm_query * norm_vectors
        norm_product[norm_product == 0] = 1e-10
        cosine_scores = (dot_product / norm_product).tolist()

        # 2. BM25 scoring
        query_tokens = tokenize(query)
        bm25_scores = [0.0] * len(self.chunks)

        if query_tokens:
            doc_count = len(self.chunks)
            doc_lens = [len(tokenize(c["text"])) for c in self.chunks]
            avg_doc_len = sum(doc_lens) / max(1, doc_count)

            # Count TFs and DFs
            df = {}
            doc_tfs = []
            for c in self.chunks:
                tokens = tokenize(c["text"])
                tf = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                doc_tfs.append(tf)
                for t in tf:
                    df[t] = df.get(t, 0) + 1

            k1 = 1.5
            b = 0.75
            for token in query_tokens:
                if token not in df:
                    continue
                df_t = df[token]
                # BM25 IDF formula
                idf = math.log((doc_count - df_t + 0.5) / (df_t + 0.5) + 1.0)

                for idx in range(doc_count):
                    tf = doc_tfs[idx].get(token, 0)
                    if tf == 0:
                        continue
                    doc_len = doc_lens[idx]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                    bm25_scores[idx] += idf * (numerator / denominator)

        # 3. Hybrid Combination
        max_bm25 = max(bm25_scores) if bm25_scores else 0
        normalized_bm25 = [s / max_bm25 for s in bm25_scores] if max_bm25 > 0 else bm25_scores
        normalized_cosine = [max(0.0, s) for s in cosine_scores]

        # Weight: 60% BM25 keyword matching, 40% dense similarity matching
        hybrid_scores = []
        for c_score, b_score in zip(normalized_cosine, normalized_bm25):
            hybrid_scores.append(0.4 * c_score + 0.6 * b_score)

        # 4. Document-level context boosting & chunk expansion
        # Find target documents with strong matches (> 0.40)
        target_docs = set()
        for idx, chunk in enumerate(self.chunks):
            if hybrid_scores[idx] > 0.40:
                target_docs.add(chunk["source"])

        # Boost all chunks of target documents. If the target document is small (<= 15 chunks),
        # mark all of its chunks for guaranteed inclusion so we parse the document in full.
        expanded_indices = set()
        for idx, chunk in enumerate(self.chunks):
            src = chunk["source"]
            if src in target_docs:
                hybrid_scores[idx] += 0.50
                doc_chunk_count = sum(1 for c in self.chunks if c["source"] == src)
                if doc_chunk_count <= 15:
                    expanded_indices.add(idx)

        sorted_indices = np.argsort(hybrid_scores)[::-1]

        # Combine top_k and expanded indices
        result_indices = []
        for idx in sorted_indices:
            if idx not in result_indices:
                result_indices.append(idx)
            if len(result_indices) >= top_k:
                break

        for idx in expanded_indices:
            if idx not in result_indices:
                result_indices.append(idx)

        results = []
        for idx in result_indices:
            result_chunk = self.chunks[idx].copy()
            result_chunk["similarity_score"] = float(cosine_scores[idx])
            result_chunk["bm25_score"] = float(bm25_scores[idx])
            result_chunk["hybrid_score"] = float(hybrid_scores[idx])
            results.append(result_chunk)

        return results

    def delete_document(self, source_filename: str) -> int:
        filtered_chunks = []
        filtered_embeddings = []
        deleted_count = 0
        for chunk, emb in zip(self.chunks, self.embeddings):
            if chunk["source"] == source_filename:
                deleted_count += 1
            else:
                filtered_chunks.append(chunk)
                filtered_embeddings.append(emb)
        self.chunks = filtered_chunks
        self.embeddings = filtered_embeddings
        if deleted_count > 0:
            self.save()
        return deleted_count

    def get_indexed_documents(self) -> dict:
        docs = {}
        for chunk in self.chunks:
            source = chunk["source"]
            if source not in docs:
                docs[source] = {"filename": source, "pages": set(), "chunk_count": 0}
            docs[source]["pages"].add(chunk["page"])
            docs[source]["chunk_count"] += 1

        result = []
        for filename, data in docs.items():
            result.append({
                "filename": filename,
                "page_count": len(data["pages"]),
                "chunk_count": data["chunk_count"]
            })
        return {"total_chunks": len(self.chunks), "documents": result}

    def save(self):
        try:
            data = {"chunks": self.chunks, "embeddings": self.embeddings}
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self.embeddings = data.get("embeddings", [])
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.chunks = []
                self.embeddings = []
