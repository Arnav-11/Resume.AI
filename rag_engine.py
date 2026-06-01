# rag_engine.py
# Full RAG pipeline for Resume.AI
#
# Embeddings   : sentence-transformers (all-MiniLM-L6-v2, CPU-friendly, ~80 MB)
# Vector DB    : In-memory cosine-similarity over a NumPy matrix
# LLM          : Mistral-7B-Instruct-v0.2 via HuggingFace Inference API
# Orchestration: LangChain RetrievalQA with a custom Mistral [INST] prompt

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-Transformers Embeddings
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
# In-memory semantic vector store
# ─────────────────────────────────────────────────────────────────────────────
class SemanticVectorStore:
    def __init__(self, docs: List[Any], embeddings: STEmbeddings):
        self.docs   = docs
        self._emb   = embeddings
        texts       = [d.page_content for d in docs]
        vecs        = embeddings.embed_documents(texts)
        self.matrix = np.array(vecs, dtype=np.float32)
        logger.info(
            f"SemanticVectorStore: {len(docs)} chunks indexed, "
            f"dim={self.matrix.shape[1]}"
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        qvec   = np.array(self._emb.embed_query(query), dtype=np.float32)
        scores = self.matrix.dot(qvec)
        top_k  = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in top_k]


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-compatible retriever (inherits BaseRetriever properly)
# ─────────────────────────────────────────────────────────────────────────────
def _make_retriever(store: SemanticVectorStore, k: int = 4):
    """
    Build a proper LangChain BaseRetriever backed by our SemanticVectorStore.
    Doing this inside a factory function avoids a top-level import that would
    crash the whole module if langchain_core is missing.
    """
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    class _SemanticRetriever(BaseRetriever):
        # Pydantic v2 — declare non-pydantic fields via model_config + __init__
        class Config:
            arbitrary_types_allowed = True

        _store: SemanticVectorStore
        _k: int

        def __init__(self, vector_store: SemanticVectorStore, top_k: int):
            super().__init__()
            # Use object.__setattr__ to bypass Pydantic field validation
            # for private attributes that aren't declared as fields.
            object.__setattr__(self, "_store", vector_store)
            object.__setattr__(self, "_k", top_k)

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:
            return self._store.similarity_search(query, k=self._k)

    return _SemanticRetriever(vector_store=store, top_k=k)


# ─────────────────────────────────────────────────────────────────────────────
# ResumeRAGEngine
# ─────────────────────────────────────────────────────────────────────────────
class ResumeRAGEngine:

    def __init__(self, hf_api_token: str):
        self.hf_api_token = hf_api_token
        self._store       = None
        self._qa_chain    = None
        self._retriever   = None
        self._ready       = False

        # Embeddings
        self._embeddings = STEmbeddings()
        if not self._embeddings.ready:
            logger.error("RAG: embeddings unavailable — check logs above")

        # Text splitter  (langchain_text_splitters is the correct package in
        # langchain ≥ 0.2; langchain.text_splitter is a deprecated shim)
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
                # Fallback: older langchain versions still export this
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

        # LLM
        self._llm = None
        if not hf_api_token:
            logger.error("RAG: HUGGINGFACE_API_TOKEN not set — LLM disabled")
        else:
            self._llm = self._load_llm(hf_api_token)

    # ── LLM loader ────────────────────────────────────────────────────────────
    # Model used for RAG — Qwen2.5-7B-Instruct is free on HuggingFace
    # inference API and supports the chat/conversational task.
    # Mistral-7B-Instruct-v0.2 was removed from the free tier in 2025.
    RAG_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def _load_llm(self, token: str):
        """
        Build the LLM using HuggingFace's InferenceClient directly via
        a custom LangChain-compatible wrapper. This bypasses langchain-huggingface
        routing issues entirely and calls the HuggingFace chat_completion API
        directly — the same API that already works in resume_analyzer.py.
        """
        try:
            from huggingface_hub import InferenceClient
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
            from langchain_core.outputs import ChatResult, ChatGeneration
            from langchain_core.callbacks import CallbackManagerForLLMRun
            from typing import Optional

            model_id = self.RAG_MODEL

            class _HFChatLLM(BaseChatModel):
                """Thin LangChain chat model wrapper around HuggingFace InferenceClient."""
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

    # ── QA chain builder ──────────────────────────────────────────────────────
    def _build_qa_chain(self, retriever):
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough, RunnableLambda

            # ChatPromptTemplate — required for ChatHuggingFace (conversational task)
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

            # LCEL chain using ChatHuggingFace
            chain = (
                {
                    "context":  retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | PROMPT
                | self._llm
                | StrOutputParser()
            )

            logger.info("RAG: LCEL chain built successfully (ChatPromptTemplate)")
            return chain

        except Exception as e:
            logger.error(f"RAG: failed to build QA chain — {e}", exc_info=True)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _clean_answer(raw: str) -> str:
        """Strip Mistral boilerplate that sometimes leaks into the result."""
        # Remove everything up to and including the last [/INST] tag
        if "[/INST]" in raw:
            raw = raw.split("[/INST]")[-1]
        # Strip common prefixes
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
            logger.error(
                "RAG build_vectorstore: embeddings not ready — "
                "install sentence-transformers and torch"
            )
            return False

        if self._splitter is None:
            logger.error(
                "RAG build_vectorstore: text splitter unavailable — "
                "install langchain-text-splitters"
            )
            return False

        try:
            from langchain_core.documents import Document

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
                logger.error("RAG build_vectorstore: 0 chunks produced — is resume_text empty?")
                return False

            self._store     = SemanticVectorStore(docs, self._embeddings)
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
            logger.info(f"RAG: ready — {len(docs)} chunks indexed, mode={mode}")
            return True

        except Exception as e:
            logger.error(f"RAG build_vectorstore failed — {e}", exc_info=True)
            return False

    def ask(self, question: str) -> Dict:
        if not self._ready or self._store is None:
            return {
                "answer": "RAG engine is not ready. Call build_vectorstore() first.",
                "sources": [],
                "mode": "error",
            }

        try:
            # ── LLM path ──────────────────────────────────────────────────────
            if self._qa_chain is not None:
                # LCEL chain returns a plain string directly
                raw_answer = self._qa_chain.invoke(question)
                answer     = self._clean_answer(raw_answer)
                top_docs   = self._store.similarity_search(question, k=4)
                sources    = self._format_sources(top_docs)
                return {"answer": answer, "sources": sources, "mode": "llm"}

            # ── Extractive fallback ───────────────────────────────────────────
            top_docs = self._store.similarity_search(question, k=4)
            sources  = self._format_sources(top_docs)
            answer   = top_docs[0].page_content.strip() if top_docs else "No relevant content found."
            return {"answer": answer, "sources": sources, "mode": "extractive"}

        except Exception as e:
            logger.error(f"RAG ask() failed — {e}", exc_info=True)
            return {"answer": "An error occurred. Please try again.", "sources": [], "mode": "error"}

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