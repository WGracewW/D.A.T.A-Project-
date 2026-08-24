"""LangGraph state schema shared by every node."""
from typing import Annotated

from typing_extensions import TypedDict, List
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from models import Document
from search.vector_store import VectorStore


class GraphState(TypedDict):
    intro: str
    guidebook_fp: str
    guide: str
    question: str
    few_shots: str
    chats_dir: str
    augmented_question: str
    context: List[Document]
    output: str
    messages: Annotated[list[AnyMessage], add_messages]
    corpus_store: VectorStore
    summary: Document
    title_page: Document
    corrected_output: str
    final_input: str
    retrieved_pages: dict
    debugging: bool
    keywords: list[str]
