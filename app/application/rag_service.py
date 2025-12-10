from typing import Sequence

from app.domain.interfaces.qa_pair_repository import QaPairRepository
from app.domain.interfaces.embedding_provider import EmbeddingProvider
from app.domain.interfaces.llm_client import LlmClient
from app.domain.models.qa_pair import QaPair
from loguru import logger


class RagService:

    def __init__(
        self,
        qa_repo: QaPairRepository,
        embedding_provider: EmbeddingProvider,
        llm_client: LlmClient,
        top_k: int = 5,
    ) -> None:

        self._qa_repo = qa_repo
        self._embeddings = embedding_provider
        self._llm = llm_client
        self._top_k = top_k

    def _build_context(self, qa_pairs: Sequence[QaPair]) -> str:
        parts: list[str] = []
        for idx, qa in enumerate(qa_pairs, start=1):
            part = f"[Q{idx}] Вопрос: {qa.question}\n" f"[A{idx}] Ответ: {qa.answer}\n"
            parts.append(part)
        return "\n".join(parts)

    def _build_prompt(self, question: str, context_qas: Sequence[QaPair]) -> str:
        context_text = self._build_context(context_qas)

        prompt = (
            "Ниже дан контекст (вопросы и ответы для абитуриентов МИРЭА).\n"
            "Используй его, чтобы ответить на вопрос пользователя.\n"
            "Если точного ответа в контексте нет, скажи, что ответить "
            "нельзя, так как ты не обладаешь такой информацией.\n\n"
            f"КОНТЕКСТ:\n{context_text}\n\n"
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}\n\n"
            "ОТВЕТ:"
        )
        return prompt

    def answer(self, question: str) -> str:
        logger.info(
            "RAG pipeline started (question_len={}, top_k={})",
            len(question),
            self._top_k,
        )
        query_vec = self._embeddings.embed(question) # 1
        logger.debug("Embedding generated (dimension={})", len(query_vec))

        context_qas = self._qa_repo.find_top_k(query_vec, k=self._top_k) # 2
        logger.debug("Top-k retrieval completed (items={})", len(context_qas))

        prompt = self._build_prompt(question, context_qas) # 3

        answer = self._llm.generate(prompt) # 4
        # lang + chain
        # _embeddings.embed | _qa_repo.find_top_k(query_vec, k=self._top_k) | _build_prompt(question, context_qas) | _llm.generate(prompt)
        # RunnableAlpha(...)
        # - llm.ainvoke(prompt) -> answer:
        '''
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
        '''
        
        
        logger.info(
            "RAG answer produced (question_len={}, context_pairs={}, answer_len={})",
            len(question),
            len(context_qas),
            len(answer),
        )
        return answer
