"""Orchestrates a run: for each PDF, build its corpus/corpora and a
VectorStore ONCE, then loop through every question (Q1...Q11) reusing that
same store — only `search()` runs per question, never re-indexing.
"""
import gc
import os
import time

from config import llm
from document_processor import build_corpora
from search.vector_store import VectorStore


def _write_outputs(chats_dir, run_store_name, response_store_name, question, response, debugging):
    with open(os.path.join(chats_dir, run_store_name), 'a', encoding='utf-8') as run:
        records = [message.pretty_repr() for message in response['messages']]
        for r in records:
            run.write(f"\n{r}")
        if debugging:
            retrieved_pages = response['retrieved_pages']
            run.write(f'Number of Pages Retrieved: {retrieved_pages["length"]}')
            run.write(f'Page Numbers: {retrieved_pages["page numbers"]}')
        run.write("\n" + "=" * 80)

    with open(os.path.join(chats_dir, response_store_name), 'a', encoding='utf-8') as resp:
        resp.write(f"\nQuestion: {question}")
        resp.write(f"\n{response['corrected_output'].pretty_repr()}")
        resp.write("\n" + "=" * 80)
        if debugging:
            retrieved_pages = response['retrieved_pages']
            resp.write(f'Number of Pages Retrieved: {retrieved_pages["length"]}')
            resp.write(f'Page Numbers: {retrieved_pages["page numbers"]}')


def _ask_all_questions(
    graph, store, summary, title_page, inputs, intro, few_shots, handbook_dir, chats_dir,
    run_store_name, response_store_name, debugging, idx_s,
):
    for idx, q in enumerate(inputs):
        question = q[1][0]
        keywords = q[1][1] if len(q[1][1]) >= 1 else None

        start_time = time.time()

        response = graph.invoke({
            'intro': intro,
            'few_shots': few_shots,
            'guidebook_fp': handbook_dir,
            'guide': None,
            'question': question,
            'augmented_question': None,
            'context': [],
            'output': None,
            'chats_dir': chats_dir,
            'messages': [],
            'corpus_store': store,
            'summary': summary,
            'title_page': title_page,
            'corrected_output': None,
            'retrieved_pages': None,
            'debugging': debugging,
            'keywords': keywords,
        })
        gc.collect()

        _write_outputs(chats_dir, run_store_name, response_store_name, question, response, debugging)

        duration = (time.time() - start_time) / 60
        print(f'\nQuestion {idx + 1} of Study {idx_s + 1} complete, Time took: {duration:.2f} minutes.')
        try:
            llm.client.close()
        except Exception:
            pass


def gen_run_9_4(
    inputs: list,
    intro: str | None,
    few_shots: str | None,
    handbook_dir: str,
    pdf_dir: str,
    chats_dir: str,
    run_number: int,
    graph,
    target_high_level: list | None = None,
    negative_titles: list | None = None,
    embedding_model_fp: str = r".\embeddings_local\all-MiniLM-L6-v2",
    debugging: bool = True,
    split: bool = True,
    target_study_words: list[str] | None = None,
    store_split_results: bool = False,
    store_fp: str | None = None,
    minimum_page_for_split: int = 80,
):
    """
    For version 9.4.

    inputs: list. [ question_id, ["query", [keywords]] ], ... ] where [keywords] = [ ['XOR','Literal','genotoxicity','rat','rats'] , ... ]
    intro: introductory line appended to every input.
    few_shots: few-shot examples appended to the end of every input, before the context.
    handbook_dir: filepath for the student handbook.
    pdf_dir: folder of studies (pdfs) to process.
    chats_dir: folder to store outputs.
    run_number: the run number.
    graph: the compiled LangGraph pipeline.
    target_high_level: sections to keep for integrated studies (e.g. toxicity, environmental, residue...).
        Any subsections under the target high level sections are used; others are ignored.
    negative_titles: sections to eliminate first when matching.
    embedding_model_fp: filepath for the local embedding model.
    debugging: bool.
    split: whether to split long PDFs into per-study chunks before indexing.
    target_study_words: only keep split chunks whose title page contains one of these words (OR condition).
    store_split_results / store_fp: optionally persist split PDF chunks to disk.
    minimum_page_for_split: minimum page count before a PDF is considered for splitting.
    """
    all_studies = os.listdir(pdf_dir)

    for idx_s, study in enumerate(all_studies):
        study_fp = os.path.join(pdf_dir, study)
        start_time_s = time.time()

        # ---- A. Build index once per PDF (document_processor.py) ----------
        corpora = build_corpora(
            study_fp,
            split=split,
            minimum_page_for_split=minimum_page_for_split,
            target_study_words=target_study_words,
            store_split_results=store_split_results,
            store_fp=store_fp,
            target_high_level=target_high_level,
            negative_titles=negative_titles,
        )

        for corpus in corpora:
            label_suffix = f"_{corpus['label']}" if corpus['label'] else ""
            run_store_name = f"{study}{label_suffix}_run_{run_number}_full_convo.txt"
            response_store_name = f"{study}{label_suffix}_run_{run_number}_response_only.txt"

            if run_store_name in os.listdir(chats_dir):  # response already stored.
                print(f"Response already recorded for {run_store_name}. Skipping.")
                continue

            store = VectorStore(model_dir=embedding_model_fp)
            store.add_documents(documents=corpus['docs'])

            with open(os.path.join(chats_dir, run_store_name), 'a', encoding='utf-8') as init_r:
                init_r.write(f"Complete Message History for PDF [{study}{label_suffix}]")

            with open(os.path.join(chats_dir, response_store_name), 'a', encoding='utf-8') as init:
                init.write(f"Responses Only for PDF [{study}{label_suffix}]")

            # ---- B. Answer all questions using the same VectorStore --------
            _ask_all_questions(
                graph, store, corpus['summary'], corpus['title_page'],
                inputs, intro, few_shots, handbook_dir, chats_dir,
                run_store_name, response_store_name, debugging, idx_s,
            )

            duration_s = (time.time() - start_time_s) / 60
            print(f"Study {idx_s + 1}{label_suffix} complete.\n Time took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")
            with open(os.path.join(chats_dir, response_store_name), 'a', encoding='utf-8') as times:
                times.write(f"\nStudy {idx_s + 1}{label_suffix} complete.\n Time took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")
