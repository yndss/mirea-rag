from typing import Sequence

from app.domain.interfaces.qa_pair_repository import QaPairRepository
from app.domain.interfaces.embedding_provider import EmbeddingProvider
from app.domain.interfaces.llm_client import LlmClient
from app.domain.models.qa_pair import QaPair
from app.infrastructure.config import RAG_MIN_SIMILARITY, RAG_QA_PROMPT_NAME, RAG_TOP_K
from app.prompts import load_prompt
from loguru import logger


class RagService:

    def __init__(
        self,
        qa_repo: QaPairRepository,
        embedding_provider: EmbeddingProvider,
        llm_client: LlmClient,
        top_k: int = RAG_TOP_K,
        qa_prompt_name: str = RAG_QA_PROMPT_NAME,
        min_similarity: float = RAG_MIN_SIMILARITY,
    ) -> None:

        self._qa_repo = qa_repo
        self._embeddings = embedding_provider
        self._llm = llm_client
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._qa_prompt_template = load_prompt(qa_prompt_name)

    def _build_context(self, qa_pairs: Sequence[QaPair]) -> str:
        parts: list[str] = []
        for idx, qa in enumerate(qa_pairs, start=1):
            part = f"- Q{idx}: {qa.question}\n" f"  A{idx}: {qa.answer}"
            parts.append(part)
        return "\n\n".join(parts)

    def _build_prompt(
        self,
        question: str,
        context_qas: Sequence[QaPair],
    ) -> str:
        context_text = self._build_context(context_qas)
        logger.info("RAG prompt context built:\n{}", context_text)

        return self._qa_prompt_template.replace("{{context}}", context_text).replace(
            "{{user_question}}", question
        )

    async def answer(self, question: str) -> str:
        logger.info(
            "RAG pipeline started (question_len={}, top_k={})",
            len(question),
            self._top_k,
        )
        query_vec = await self._embeddings.embed(question)  # 1
        logger.debug("Embedding generated (dimension={})", len(query_vec))

        context_qas = await self._qa_repo.find_top_k(
            query_vec,
            self._top_k,
            min_similarity=self._min_similarity,
        )  # 2
        logger.debug("Top-k retrieval completed (items={})", len(context_qas))

        prompt = self._build_prompt(question, context_qas)  # 3

        answer = await self._llm.generate(prompt)  # 4
        logger.info("RAG full answer:\n{}", answer)
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
