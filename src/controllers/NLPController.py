from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from AI.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import logging
import os
import asyncio
from sentence_transformers import CrossEncoder
class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)
        self._cross_encoder = None  # lazy-loaded for reranking; load once, reuse

    def _get_cross_encoder(self):
        """Load cross-encoder once and reuse. Suppress loading logs and progress bar."""
        if self._cross_encoder is None:
            # Suppress loading logs and tqdm progress bar
            tqdm_orig = os.environ.get("TQDM_DISABLE")
            tf_verbosity_orig = os.environ.get("TRANSFORMERS_VERBOSITY")
            os.environ["TQDM_DISABLE"] = "1"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            loggers = ("transformers", "sentence_transformers", "httpx")
            levels = {name: logging.getLogger(name).level for name in loggers}
            for name in loggers:
                logging.getLogger(name).setLevel(logging.ERROR)
            try:
                import transformers  # noqa: F401
                transformers.logging.set_verbosity_error()
            except Exception:
                pass
            try:
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            finally:
                for name in loggers:
                    logging.getLogger(name).setLevel(levels.get(name, logging.NOTSET))
                if tqdm_orig is None:
                    os.environ.pop("TQDM_DISABLE", None)
                else:
                    os.environ["TQDM_DISABLE"] = tqdm_orig
                if tf_verbosity_orig is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = tf_verbosity_orig
        return self._cross_encoder

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        
        vectors = await self.embedding_client.embed_texts_batch(
            texts=texts,
            document_type=DocumentTypeEnum.DOCUMENT.value,
            batch_size=100  # OpenAI supports up to 2048
        )
        # Fallback when batch returns None, wrong length, or contains None (e.g. provider not implementing it or API error)
        if vectors is None or len(vectors) != len(texts) or any(v is None for v in vectors):
            self.logger.warning(
                f"embed_texts_batch returned {len(vectors) if vectors else 0} vectors for {len(texts)} texts; "
                "falling back to per-text embedding."
            )
            vectors = await asyncio.gather(*[
                self.embedding_client.embed_text(text=t, document_type=DocumentTypeEnum.DOCUMENT.value)
                for t in texts
            ])
            if any(v is None for v in vectors) or len(vectors) != len(texts):
                self.logger.error(
                    f"Fallback embedding failed: got {len(vectors)} vectors ({sum(1 for v in vectors if v is None)} None) for {len(texts)} texts."
                )
                return False

        # step3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: get text embedding vector
        vector = await self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        # step3: do semantic search
        results = self.vectordb_client.hybrid_search(
            collection_name=collection_name,
            vector=vector,
            query_text=text,
            limit=limit
        )

        if not results:
            return False

        return results
    async def rerank_documents(self, query: str, documents: List, top_k: int = 5):
        """Rerank retrieved documents using cross-encoder (model loaded once, then reused)."""
        model = self._get_cross_encoder()
        pairs = [[query, doc.text] for doc in documents]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, model.predict, pairs)
        
        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
    
    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        # Check cache first
        if hasattr(self, 'cache'):
            cached_answer = self.cache.get_answer(query, project.project_id)
            if cached_answer:
                self.logger.info("Returning cached answer")
                return cached_answer, None, None
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            self.logger.warning(f"No documents retrieved for query: {query}")
            return answer, full_prompt, chat_history
        
        self.logger.info(f"Retrieved {len(retrieved_documents)} documents")
        # Add reranking
        retrieved_documents = await self.rerank_documents(
            query=query,
            documents=retrieved_documents,
            top_k=min(5, len(retrieved_documents))
        )
        self.logger.info(f"Reranked to {len(retrieved_documents)} documents")
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text,
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # step3: Construct Generation Client Prompts
        try:
            chat_history = [
                await self.generation_client.construct_prompt(
                    prompt=system_prompt,
                    role=self.generation_client.enums.SYSTEM.value,
                )
            ]
            self.logger.info("Chat history constructed successfully")
        except Exception as e:
            self.logger.error(f"Error constructing chat history: {str(e)}")
            return answer, full_prompt, chat_history

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        try:
            self.logger.info(f"Calling generate_text with model: {self.generation_client.generation_model_id}")
            answer = await self.generation_client.generate_text(
                prompt=full_prompt,
                chat_history=chat_history
            )
            if answer:
                self.logger.info("Answer generated successfully")
            else:
                self.logger.error("generate_text returned None or empty answer")
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            return answer, full_prompt, chat_history
        # Cache the answer
        if answer and hasattr(self, 'cache'):
            self.cache.set_answer(query, project.project_id, answer)
        return answer, full_prompt, chat_history

    async def answer_rag_question_stream(self, project: Project, query: str, limit: int = 10):
        """Stream the answer to a RAG question."""
        # Check cache first (streaming cache returned as single block for simplicity or not cached)
        if hasattr(self, 'cache'):
            cached_answer = self.cache.get_answer(query, project.project_id)
            if cached_answer:
                self.logger.info("Returning cached answer (non-streaming)")
                yield cached_answer
                return

        # step1: retrieve related documents
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            self.logger.warning(f"No documents retrieved for query: {query}")
            yield "I couldn't find any relevant information to answer your question."
            return
        
        # Add reranking
        retrieved_documents = await self.rerank_documents(
            query=query,
            documents=retrieved_documents,
            top_k=min(5, len(retrieved_documents))
        )
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text,
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # step3: Construct Generation Client Prompts
        chat_history = [
            await self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer in streaming mode
        full_answer = []
        async for chunk in self.generation_client.generate_text_stream(
            prompt=full_prompt,
            chat_history=chat_history
        ):
            full_answer.append(chunk)
            yield chunk

        # Cache the full answer at the end
        if full_answer and hasattr(self, 'cache'):
            self.cache.set_answer(query, project.project_id, "".join(full_answer))
