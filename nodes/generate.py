"""Calls the LLM to produce a raw answer for the augmented prompt."""
from config import llm
from state import GraphState
from utils import clean_prompt_input


def generate(state:GraphState):

    if state['debugging'] == True:
        print("Generating...")

    augmented_input = state['augmented_question']
    # debugging
    print("Cleaning Prompt Input...")
    final_input = clean_prompt_input(augmented_input)

    output = llm.invoke(final_input)

    return{
        'output':output,
        'final_input':final_input
    }
