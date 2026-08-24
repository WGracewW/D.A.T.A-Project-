"""Entry point. Wires up config, questions, and the compiled graph, then
runs the pipeline over every PDF in pdf_dir."""
from config import chats_dir, handbook_dir, pdf_dir, run_number
from graph import graph
from questions import few_shots, inputs, intro
from run_methods import gen_run_9_4

if __name__ == "__main__":
    gen_run_9_4(
        inputs=inputs,
        intro=intro,
        few_shots=few_shots,
        handbook_dir=handbook_dir,
        pdf_dir=pdf_dir,
        chats_dir=chats_dir,
        run_number=run_number,
        graph=graph,
        debugging=True,
        split=False,
        target_study_words=None,
        store_split_results=False,
        store_fp=None,
        minimum_page_for_split=100,
    )
