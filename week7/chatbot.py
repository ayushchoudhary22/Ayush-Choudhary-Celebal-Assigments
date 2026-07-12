import re
from vectorstore import tokenize, get_fallback_embedding, SimpleVectorStore

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def get_embedding(text: str, api_key: str = None) -> list[float]:
    """Generate embedding for a query, using API if available."""
    if api_key and HAS_GENAI:
        try:
            genai.configure(api_key=api_key)
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return response['embedding']
        except Exception as e:
            print(f"API embedding failed, using local: {e}")
    return get_fallback_embedding(text)


def _extract_sentences(text: str) -> list[str]:
    """Split text into deduplicated sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    seen = set()
    for s in sentences:
        s = s.strip()
        normalized = s.lower().strip('., ')
        if len(s) > 25 and normalized not in seen:
            seen.add(normalized)
            result.append(s)
    return result


CATEGORIES = {
    "projects": ["projects", "project", "portfolio"],
    "experience": ["experience", "job", "work", "intern", "employment", "history"],
    "education": ["education", "university", "college", "school", "degree", "b.tech", "study"],
    "skills": ["skills", "languages", "technologies", "tools", "database", "stack", "programming"],
    "contact": ["contact", "email", "phone", "address", "github", "linkedin"]
}


def is_section_header(unit_lower: str, category: str) -> bool:
    """Detects if a line represents the start of a resume section heading."""
    synonyms = CATEGORIES[category]
    prefixes = ["technical", "academic", "key", "my", "professional", "work", "personal"]
    words = unit_lower.split()
    if not words:
        return False
    first_word = words[0].strip(":,.-_")
    first_two = " ".join(w.strip(":,.-_") for w in words[:2])
    for syn in synonyms:
        if syn == first_word or syn == first_two:
            return True
        for pref in prefixes:
            if first_word == pref and len(words) > 1 and syn == words[1].strip(":,.-_"):
                return True
    return False


# Section headers to split on (appear at start of a word boundary)
_SECTION_SPLIT_RE = re.compile(
    r'(?<=[\s,;])'
    r'(?=(?:Education|Experience|Projects|Technical\s+Skills|Skills|Certifications|Contact|Summary|Objective)'
    r'(?:\s|$|\:))',
    re.IGNORECASE
)


def split_into_units(text: str) -> list[str]:
    """Splits a document text into semantic units.

    1. Replaces bullet symbols with newlines.
    2. Splits on newlines.
    3. Within each segment, splits AGAIN at section heading keywords that the
       PDF extractor may have concatenated together (e.g. "Education ... Experience ...").
    4. Further splits on sentence boundaries.
    """
    for b in ['\u2022', '\u25cf', '\u25cb', '\u25aa', '\u25ab', '\u25b6', '\u25c6', '•']:
        text = text.replace(b, '\n')
    lines = text.split('\n')
    units = []
    for line in lines:
        # Split at section headings that may be concatenated mid-line
        sub_parts = _SECTION_SPLIT_RE.split(line)
        for p in sub_parts:
            # Further split on sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', p)
            for s in sentences:
                s_clean = s.strip()
                # Skip very short fragments and lines starting lowercase
                # (PDF page-wrap artifacts like 'tions using...' or 'anning, dietary...')
                if s_clean and len(s_clean) >= 20 and not re.match(r'^[a-z]', s_clean):
                    units.append(s_clean)
    return units


def synthesize_answer_offline(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Offline QA: If query matches a specific profile category (projects, education, etc.),
    extracts and formats ONLY that section. Otherwise, displays the top relevant document chunks.
    """
    if not retrieved_chunks:
        return ("I could not find any relevant information in the indexed documents. "
                "Try rephrasing your question or make sure relevant documents are indexed.")

    # 1. Detect target category from query
    query_lower = query.lower()
    query_tokens = tokenize(query_lower)
    target_category = None
    for cat, synonyms in CATEGORIES.items():
        for syn in synonyms:
            if syn in query_tokens:
                target_category = cat
                break
        if target_category:
            break

    # 2. If target category is found, perform targeted section extraction
    if target_category:
        docs_chunks = {}
        for chunk in retrieved_chunks:
            src = chunk['source']
            if src not in docs_chunks:
                docs_chunks[src] = []
            docs_chunks[src].append(chunk)

        answer_parts = []
        sources_used = set()

        for src, chunks in docs_chunks.items():
            # Sort chunks by order in document
            chunks.sort(key=lambda x: (x['page'], x.get('chunk_index', 0)))
            
            # Merge and clean text
            merged_text = " ".join(c['text'] for c in chunks)
            merged_text = merged_text.replace('\ufb01', 'fi').replace('\ufb02', 'fl').replace('\u0131', 'i')
            
            units = split_into_units(merged_text)
            extracted_units = []
            flow_score = 0.0
            
            other_categories = {cat: syns for cat, syns in CATEGORIES.items() if cat != target_category}
            
            for unit in units:
                unit_lower = unit.lower()
                
                # Stop flow if we hit a different category header
                stop_flow = False
                for other_cat in other_categories:
                    if is_section_header(unit_lower, other_cat):
                        stop_flow = True
                        break
                        
                if stop_flow:
                    flow_score = 0.0
                    continue
                
                # Start flow if we hit the target category keyword
                start_flow = is_section_header(unit_lower, target_category)
                        
                if start_flow:
                    flow_score = 3.0
                    
                if flow_score > 0.15:
                    sources_used.add(f"{src} (Page {chunks[0]['page']})")
                    # Format lines containing pipes |
                    if '|' in unit:
                        parts = unit.split('|')
                        title = parts[0].strip()
                        details = ' | '.join(p.strip() for p in parts[1:])
                        extracted_units.append(f"**{title}** ({details})")
                    else:
                        extracted_units.append(unit)
                    
                    # Decay flow score forward
                    flow_score *= 0.85
                    
            if extracted_units:
                section_md = f"### 📄 From **{src}** (Page {chunks[0]['page']})\n"
                for u in extracted_units:
                    if u.startswith('**') and u.endswith(')'):
                        section_md += f"\n### {u}\n"
                    else:
                        section_md += f"- {u}\n"
                answer_parts.append(section_md)

        if answer_parts:
            answer = f"**Here are the {target_category} details found in your documents:**\n\n"
            answer += "\n".join(answer_parts)
            answer += "\n---\n📚 **Sources:** " + " | ".join(sorted(sources_used))
            return answer

    # 3. Fallback: Display raw formatted chunks (best for textbook/notes Q&A)
    retrieved_chunks.sort(key=lambda x: x.get('hybrid_score', x.get('similarity_score', 0)), reverse=True)
    relevant_chunks = [c for c in retrieved_chunks if c.get('hybrid_score', c.get('similarity_score', 0)) > 0.1]
    if not relevant_chunks:
        relevant_chunks = retrieved_chunks[:2]
    relevant_chunks = relevant_chunks[:3]

    answer_parts = []
    sources_used = set()

    for idx, chunk in enumerate(relevant_chunks):
        src = chunk['source']
        page = chunk['page']
        score = chunk.get('hybrid_score', chunk.get('similarity_score', 0))
        sources_used.add(f"{src} (Page {page})")

        text = chunk['text']
        text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl').replace('\u0131', 'i')

        bullets = re.split(r'[\u2022\u25cf\u25cb\u25aa\u25ab\u25b6\u25c6]', text)
        formatted_bullets = []

        for b in bullets:
            b_clean = b.strip()
            if not b_clean:
                continue
            if '|' in b_clean:
                parts = b_clean.split('|')
                title = parts[0].strip()
                details = ' | '.join(p.strip() for p in parts[1:])
                formatted_bullets.append(f"**{title}** ({details})")
            else:
                formatted_bullets.append(b_clean)

        chunk_md = f"### 📄 From **{src}** (Page {page}, Match: {score*100:.0f}%)\n"
        for bullet in formatted_bullets:
            chunk_md += f"- {bullet}\n"
        
        answer_parts.append(chunk_md)

    answer = "**Based on your documents, here is what I found:**\n\n"
    answer += "\n".join(answer_parts)
    answer += "\n---\n📚 **Sources:** " + " | ".join(sorted(sources_used))
    return answer


def generate_llm_answer(query: str, retrieved_chunks: list[dict], api_key: str = None) -> str:
    """
    Generate answer from retrieved chunks.
    Uses Gemini API if api_key provided; otherwise uses extractive QA.
    """
    if not retrieved_chunks:
        return ("No relevant information found in the indexed documents. "
                "Please make sure documents are indexed first.")

    # Build context
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']} (Page {chunk['page']})\n"
            f"Content: {chunk['text']}\n"
        )
    context = "\n---\n".join(context_parts)

    # Try API
    if api_key and HAS_GENAI:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are an expert AI assistant. Answer the user's question based strictly on the "
                "provided document context. Cite sources inline. If the context is insufficient, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\n\nDetailed Answer:"
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")

    return synthesize_answer_offline(query, retrieved_chunks)


def query_rag(question: str, vector_store: SimpleVectorStore,
              api_key: str = None, top_k: int = 4) -> dict:
    """
    Full RAG pipeline: embed query -> retrieve -> generate answer.
    Returns dict with answer, retrieved chunks, and mode.
    """
    query_embedding = get_embedding(question, api_key)
    retrieved = vector_store.similarity_search(query_embedding, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No documents indexed yet. Please upload and index documents first.",
            "sources": [],
            "mode": "no_data"
        }

    answer = generate_llm_answer(question, retrieved, api_key)
    mode = "neural" if (api_key and HAS_GENAI) else "local"

    sources = []
    for chunk in retrieved:
        sources.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "score": chunk.get("similarity_score", 0),
            "preview": chunk["text"][:200]
        })

    return {"answer": answer, "sources": sources, "mode": mode}
