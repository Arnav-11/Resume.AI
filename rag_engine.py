# rag_engine.py
# Full RAG pipeline for Resume.AI
#
# Embeddings   : sentence-transformers (all-MiniLM-L6-v2, CPU-friendly, ~80 MB)
# Vector DB    : ChromaDB (persistent on disk, replaces in-memory NumPy matrix)
# LLM          : Qwen2.5-7B-Instruct via HuggingFace Inference API
# Orchestration: LangChain LCEL chain with a custom HF chat wrapper

import os
import uuid
import logging
import numpy as np
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Sentence-Transformers Embeddings
# (unchanged — ChromaDB uses this same class for embedding)
# ─────────────────────────────────────────────────────────────────────────────
class STEmbeddings:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            logger.info(f"STEmbeddings: loaded {self.MODEL_NAME}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed.\n"
                "Fix: pip install sentence-transformers torch"
            )
        except Exception as e:
            logger.error(f"STEmbeddings: failed to load — {e}")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Vector Store
# CHANGED: replaces the old SemanticVectorStore (NumPy matrix)
#
# What changed and why:
#   Old: np.array stored in RAM — lost on every app restart
#   New: ChromaDB collection stored on disk at ./chroma_db/
#        Survives restarts, supports larger datasets, proper vector DB
# ─────────────────────────────────────────────────────────────────────────────
class ChromaVectorStore:
    # Each resume session gets a unique collection name so sessions
    # don't bleed into each other.
    CHROMA_DIR = "./chroma_db"

    def __init__(self, docs: List[Any], embeddings: STEmbeddings):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed.\n"
                "Fix: pip install chromadb"
            )

        self._emb = embeddings

        # PersistentClient saves the collection to disk at CHROMA_DIR.
        # Old code: np.array(...) — all in RAM, gone on restart.
        # New code: chromadb.PersistentClient — saved to disk automatically.
        self._client = chromadb.PersistentClient(path=self.CHROMA_DIR)

        # Use a unique collection name per session so multiple users
        # don't share/overwrite each other's resume data.
        self._collection_name = f"resume_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.create_collection(
            name=self._collection_name,
            # cosine is the right metric for normalized sentence-transformer vectors
            metadata={"hnsw:space": "cosine"},
        )

        # Add all chunks to ChromaDB
        # Old: matrix = np.array(vecs) → just stored numbers
        # New: collection.add() stores text, embedding, metadata, and id together
        texts     = [d.page_content for d in docs]
        embeddings_list = embeddings.embed_documents(texts)
        metadatas = [d.metadata for d in docs]
        ids       = [f"chunk_{i}" for i in range(len(docs))]

        self._collection.add(
            documents=texts,           # the raw text of each chunk
            embeddings=embeddings_list, # the vector for each chunk
            metadatas=metadatas,       # {"source": "resume"} or {"source": "job_description"}
            ids=ids,                   # unique ID for each chunk
        )

        # Keep original docs so we can return LangChain Document objects
        self._docs = docs

        logger.info(
            f"ChromaVectorStore: {len(docs)} chunks indexed in "
            f"collection '{self._collection_name}' at {self.CHROMA_DIR}"
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        # Old code:
        #   qvec   = np.array(self._emb.embed_query(query))
        #   scores = self.matrix.dot(qvec)           # brute force dot product
        #   top_k  = np.argsort(scores)[::-1][:k]
        #   return [self.docs[i] for i in top_k]
        #
        # New code:
        #   ChromaDB handles the similarity search internally using HNSW index.
        #   HNSW (Hierarchical Navigable Small World) is an approximate nearest
        #   neighbor algorithm — much faster than brute force for large datasets.

        query_embedding = self._emb.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        # ChromaDB returns results as lists of lists — one list per query.
        # We sent one query so we take index [0].
        from langchain_core.documents import Document
        docs = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def cleanup(self):
        """Delete the collection when the session is done to free disk space."""
        try:
            self._client.delete_collection(self._collection_name)
            logger.info(f"ChromaVectorStore: cleaned up collection '{self._collection_name}'")
        except Exception as e:
            logger.warning(f"ChromaVectorStore: cleanup failed — {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-compatible retriever
# UNCHANGED in logic — just now backed by ChromaVectorStore instead of
# SemanticVectorStore, but the interface is identical.
# ─────────────────────────────────────────────────────────────────────────────
def _make_retriever(store: ChromaVectorStore, k: int = 4):
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    class _ChromaRetriever(BaseRetriever):
        class Config:
            arbitrary_types_allowed = True

        _store: ChromaVectorStore
        _k: int

        def __init__(self, vector_store: ChromaVectorStore, top_k: int):
            super().__init__()
            object.__setattr__(self, "_store", vector_store)
            object.__setattr__(self, "_k", top_k)

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:
            return self._store.similarity_search(query, k=self._k)

    return _ChromaRetriever(vector_store=store, top_k=k)


# ─────────────────────────────────────────────────────────────────────────────
# ResumeRAGEngine — main coordinator
# Changes from old version:
#   1. self._store is now ChromaVectorStore instead of SemanticVectorStore
#   2. build_vectorstore() calls ChromaVectorStore(...) instead of SemanticVectorStore(...)
#   3. Added cleanup() call to delete ChromaDB collection after use
#   4. Everything else (LLM, chain, ask, helpers) is UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
class ResumeRAGEngine:

    def __init__(self, hf_api_token: str):
        self.hf_api_token = hf_api_token
        self._store       = None   # will be ChromaVectorStore
        self._qa_chain    = None
        self._retriever   = None
        self._ready       = False

        # Embeddings — unchanged
        self._embeddings = STEmbeddings()
        if not self._embeddings.ready:
            logger.error("RAG: embeddings unavailable — check logs above")

        # Text splitter — unchanged
        self._splitter = None
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=80,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            logger.info("RAG: text splitter ready (langchain_text_splitters)")
        except ImportError:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                self._splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                logger.info("RAG: text splitter ready (langchain.text_splitter fallback)")
            except ImportError:
                logger.error(
                    "RAG: text splitter unavailable.\n"
                    "Fix: pip install langchain-text-splitters"
                )

        # LLM — unchanged
        self._llm = None
        if not hf_api_token:
            logger.error("RAG: HUGGINGFACE_API_TOKEN not set — LLM disabled")
        else:
            self._llm = self._load_llm(hf_api_token)

    # ── LLM loader — UNCHANGED ────────────────────────────────────────────────
    RAG_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def _load_llm(self, token: str):
        try:
            from huggingface_hub import InferenceClient
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
            from langchain_core.outputs import ChatResult, ChatGeneration
            from langchain_core.callbacks import CallbackManagerForLLMRun
            from typing import Optional

            model_id = self.RAG_MODEL

            class _HFChatLLM(BaseChatModel):
                client: object
                model_id: str

                class Config:
                    arbitrary_types_allowed = True

                @property
                def _llm_type(self) -> str:
                    return "huggingface_chat"

                def _generate(
                    self,
                    messages: list[BaseMessage],
                    stop=None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs,
                ) -> ChatResult:
                    hf_msgs = []
                    for m in messages:
                        if isinstance(m, SystemMessage):
                            hf_msgs.append({"role": "system", "content": m.content})
                        elif isinstance(m, HumanMessage):
                            hf_msgs.append({"role": "user", "content": m.content})
                        elif isinstance(m, AIMessage):
                            hf_msgs.append({"role": "assistant", "content": m.content})
                        else:
                            hf_msgs.append({"role": "user", "content": m.content})

                    response = self.client.chat_completion(
                        messages=hf_msgs,
                        model=self.model_id,
                        max_tokens=600,
                        temperature=0.3,
                    )
                    text = response.choices[0].message.content or ""
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

            hf_client = InferenceClient(api_key=token)
            llm = _HFChatLLM(client=hf_client, model_id=model_id)
            logger.info(f"RAG: LLM ready — {model_id} via InferenceClient")
            return llm

        except ImportError as e:
            logger.error(f"RAG: missing dependency — {e}")
        except Exception as e:
            logger.error(f"RAG: LLM init failed — {e}")

        return None

    # ── QA chain builder — UNCHANGED ──────────────────────────────────────────
    def _build_qa_chain(self, retriever):
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough, RunnableLambda

            PROMPT = ChatPromptTemplate.from_messages([
                ("system",
                    "You are an expert resume coach and career advisor. "
                    "Use ONLY the resume/job-description content provided to answer the question. "
                    "Be specific — reference actual skills, roles, projects, or dates from the content. "
                    "If the information is not present, say so clearly."),
                ("human",
                    "Content:\n{context}\n\nQuestion: {question}"),
            ])

            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            chain = (
                {
                    "context":  retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | PROMPT
                | self._llm
                | StrOutputParser()
            )

            logger.info("RAG: LCEL chain built successfully")
            return chain

        except Exception as e:
            logger.error(f"RAG: failed to build QA chain — {e}", exc_info=True)
            return None

    # ── Helpers — UNCHANGED ───────────────────────────────────────────────────
    @staticmethod
    def _clean_answer(raw: str) -> str:
        if "[/INST]" in raw:
            raw = raw.split("[/INST]")[-1]
        for prefix in ("Answer:", "answer:", "ANSWER:"):
            if raw.lstrip().startswith(prefix):
                raw = raw.lstrip()[len(prefix):]
                break
        return raw.strip()

    @staticmethod
    def _format_sources(source_docs: List[Any]) -> List[Dict]:
        out = []
        for d in source_docs:
            text = d.page_content
            out.append({
                "text":   (text[:250] + "…") if len(text) > 250 else text,
                "source": d.metadata.get("source", "resume"),
            })
        return out

    # ── Public API ────────────────────────────────────────────────────────────
    def build_vectorstore(
        self, resume_text: str, job_description: Optional[str] = None
    ) -> bool:
        if not self._embeddings.ready:
            logger.error("RAG build_vectorstore: embeddings not ready")
            return False

        if self._splitter is None:
            logger.error("RAG build_vectorstore: text splitter unavailable")
            return False

        try:
            from langchain_core.documents import Document

            # CHANGED: cleanup old ChromaDB collection before building a new one.
            # Old code had no cleanup because NumPy matrix was just garbage collected.
            # ChromaDB persists to disk so we must delete the old collection ourselves.
            if self._store is not None:
                self._store.cleanup()

            docs: List[Document] = []

            for chunk in self._splitter.split_text(resume_text):
                chunk = chunk.strip()
                if chunk:
                    docs.append(Document(page_content=chunk, metadata={"source": "resume"}))

            if job_description and job_description.strip():
                for chunk in self._splitter.split_text(job_description):
                    chunk = chunk.strip()
                    if chunk:
                        docs.append(
                            Document(page_content=chunk, metadata={"source": "job_description"})
                        )

            if not docs:
                logger.error("RAG build_vectorstore: 0 chunks produced")
                return False

            # CHANGED: ChromaVectorStore instead of SemanticVectorStore
            # Old: self._store = SemanticVectorStore(docs, self._embeddings)
            # New: self._store = ChromaVectorStore(docs, self._embeddings)
            self._store     = ChromaVectorStore(docs, self._embeddings)
            self._retriever = _make_retriever(self._store, k=4)

            if self._llm is not None:
                self._qa_chain = self._build_qa_chain(self._retriever)
                if self._qa_chain is None:
                    logger.warning("RAG: QA chain build failed — falling back to extractive mode")
            else:
                logger.warning("RAG: no LLM available — using extractive fallback")
                self._qa_chain = None

            self._ready = True
            mode = "llm" if self._qa_chain else "extractive"
            logger.info(f"RAG: ready — {len(docs)} chunks indexed via ChromaDB, mode={mode}")
            return True

        except Exception as e:
            logger.error(f"RAG build_vectorstore failed — {e}", exc_info=True)
            return False

    # ── ask() — UNCHANGED ─────────────────────────────────────────────────────
    def ask(self, question: str) -> Dict:
        if not self._ready or self._store is None:
            return {
                "answer": "RAG engine is not ready. Call build_vectorstore() first.",
                "sources": [],
                "mode": "error",
            }

        try:
            if self._qa_chain is not None:
                raw_answer = self._qa_chain.invoke(question)
                answer     = self._clean_answer(raw_answer)
                top_docs   = self._store.similarity_search(question, k=4)
                sources    = self._format_sources(top_docs)
                return {"answer": answer, "sources": sources, "mode": "llm"}

            top_docs = self._store.similarity_search(question, k=4)
            sources  = self._format_sources(top_docs)
            answer   = top_docs[0].page_content.strip() if top_docs else "No relevant content found."
            return {"answer": answer, "sources": sources, "mode": "extractive"}

        except Exception as e:
            logger.error(f"RAG ask() failed — {e}", exc_info=True)
            return {"answer": "An error occurred. Please try again.", "sources": [], "mode": "error"}

    # ── Convenience wrappers — UNCHANGED ──────────────────────────────────────
    def get_targeted_feedback(self, section: str) -> str:
        result = self.ask(
            f"Looking at the '{section}' section of this resume, "
            f"give exactly 3 numbered, specific, actionable improvement suggestions. "
            f"Reference actual content from the resume in each suggestion."
        )
        return result.get("answer", "No feedback available.")

    def get_semantic_jd_match_insights(self) -> str:
        result = self.ask(
            "Compare this resume against the job description. "
            "Structure your answer in 3 parts:\n"
            "(1) Top 3 resume strengths that directly match the job requirements.\n"
            "(2) The single most important skill or experience gap.\n"
            "(3) One specific, concrete change to strengthen this resume for the role."
        )
        return result.get("answer", "No insights available.")
