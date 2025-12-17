import time
from typing import Sequence

from app.domain.interfaces.qa_pair_repository import QaPairRepository
from app.domain.interfaces.embedding_provider import EmbeddingProvider
from app.domain.interfaces.llm_client import LlmClient
from app.domain.models.qa_pair import QaPair
from app.domain.interfaces.rag_run_repository import RagRunRepository
from app.domain.models.rag_run import RagRun, RagRunHit
from app.infrastructure.config import (
    EMBEDDING_MODEL_NAME,
    OPENROUTER_MODEL_NAME,
    OPENROUTER_TEMPERATURE,
    RAG_MIN_SIMILARITY,
    RAG_QA_PROMPT_NAME,
    RAG_TOP_K,
    SYSTEM_PROMPT_NAME,
)
from app.prompts import load_prompt
from app.pricing import estimate_llm_cost_usd
from loguru import logger


class RagService:

    def __init__(
        self,
        qa_repo: QaPairRepository,
        embedding_provider: EmbeddingProvider,
        llm_client: LlmClient,
        run_repo: RagRunRepository | None = None,
        top_k: int = RAG_TOP_K,
        qa_prompt_name: str = RAG_QA_PROMPT_NAME,
        min_similarity: float = RAG_MIN_SIMILARITY,
    ) -> None:

        self._qa_repo = qa_repo
        self._embeddings = embedding_provider
        self._llm = llm_client
        self._run_repo = run_repo
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._qa_prompt_name = qa_prompt_name
        self._qa_prompt_template = load_prompt(qa_prompt_name)

    def _build_context(self, qa_pairs: Sequence[QaPair]) -> str:
        parts: list[str] = []
        for idx, qa in enumerate(qa_pairs, start=1):
            part = f"- Q{idx}: {qa.question}\n" f"  A{idx}: {qa.answer}"
            parts.append(part)
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context_text: str) -> str:
        return self._qa_prompt_template.replace("{{context}}", context_text).replace(
            "{{user_question}}", question
        )

    async def answer(self, question: str, user_id: int | None = None) -> str:
        t_total_start = time.perf_counter()
        logger.info(
            "RAG pipeline started (question_len={}, top_k={}, min_similarity={}, question={})",
            len(question),
            self._top_k,
            self._min_similarity,
            question,
        )

        t_embed_start = time.perf_counter()
        query_vec = await self._embeddings.embed(question)  # 1
        t_embed_end = time.perf_counter()
        logger.debug("Embedding generated (dimension={})", len(query_vec))

        t_retrieval_start = time.perf_counter()
        retrieved_hits = await self._qa_repo.find_top_k(
            query_vec,
            self._top_k,
        )  # 2
        t_retrieval_end = time.perf_counter()

        used_hits = [
            hit for hit in retrieved_hits if hit.similarity >= self._min_similarity
        ]
        logger.info("RAG min_similarity configured: {}", self._min_similarity)
        logger.info(
            "RAG hits used in context (used/total={} / {}): {}",
            len(used_hits),
            len(retrieved_hits),
            [hit.rank for hit in used_hits],
        )

        context_qas = [hit.qa_pair for hit in used_hits]
        context_text = self._build_context(context_qas)
        logger.info("RAG prompt context built:\n{}", context_text)

        prompt = self._build_prompt(question, context_text)  # 3
        logger.info("RAG final prompt:\n{}", prompt)

        t_llm_start = time.perf_counter()
        generation = await self._llm.generate(prompt)  # 4
        t_llm_end = time.perf_counter()

        answer = generation.text
        model_name = generation.model or (OPENROUTER_MODEL_NAME or "")
        cost_usd = estimate_llm_cost_usd(
            model_name=model_name,
            prompt_tokens=generation.usage.prompt_tokens if generation.usage else None,
            completion_tokens=(
                generation.usage.completion_tokens if generation.usage else None
            ),
        )
        logger.info("RAG full answer:\n{}", answer)

        t_total_end = time.perf_counter()

        if self._run_repo is not None:
            run = RagRun(
                id=None,
                created_at=None,
                user_id=user_id,
                question_text=question,
                retriever_top_k=self._top_k,
                similarity_threshold=self._min_similarity,
                distance_metric="cosine",
                context_text=context_text,
                final_prompt_text=prompt,
                model_name=model_name,
                temperature=OPENROUTER_TEMPERATURE,
                extra_params={
                    "system_prompt_name": SYSTEM_PROMPT_NAME,
                    "qa_prompt_name": self._qa_prompt_name,
                    "embedding_model_name": EMBEDDING_MODEL_NAME,
                },
                answer_text=answer,
                usage_prompt_tokens=(
                    generation.usage.prompt_tokens if generation.usage else None
                ),
                usage_completion_tokens=(
                    generation.usage.completion_tokens if generation.usage else None
                ),
                usage_total_tokens=(
                    generation.usage.total_tokens if generation.usage else None
                ),
                cost_usd=cost_usd,
                latency_ms_total=int((t_total_end - t_total_start) * 1000),
                latency_ms_retrieval=int((t_retrieval_end - t_retrieval_start) * 1000),
                latency_ms_llm=int((t_llm_end - t_llm_start) * 1000),
                latency_ms_embedding=int((t_embed_end - t_embed_start) * 1000),
            )

            run_hits: list[RagRunHit] = [
                RagRunHit(
                    rag_run_id=None,
                    rank=hit.rank,
                    qa_pair_id=hit.qa_pair.id,
                    distance=hit.distance,
                    similarity=hit.similarity,
                    used_in_context=(hit.similarity >= self._min_similarity),
                )
                for hit in retrieved_hits
            ]

            run_id = await self._run_repo.add_run(run, run_hits)
            logger.info("RAG run persisted (rag_run_id={})", run_id)
        # lang + chain
        # _embeddings.embed | _qa_repo.find_top_k(query_vec, k=self._top_k) | _build_prompt(question, context_qas) | _llm.generate(prompt)
        # RunnableAlpha(...)
        # - llm.ainvoke(prompt) -> answer:
        """
        llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
              )
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        """

        logger.info(
            "RAG answer produced (question_len={}, context_pairs={}, answer_len={})",
            len(question),
            len(context_qas),
            len(answer),
        )
        return answer
