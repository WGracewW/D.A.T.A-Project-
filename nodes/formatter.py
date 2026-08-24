"""Formats/normalizes the raw LLM answer into the <CATEGORY> : <ANSWER> format."""
from langchain_core.messages import AIMessage, HumanMessage

from config import llm
from state import GraphState


def formatter(state:GraphState):

    if state['debugging'] == True:
        print("Formatting...")

    output = state['output'].text()
    question = state['question']
    template = f"""
        You need to read a question and its response, then respond with only the target information from the response.
        \nThe Question:
        \n{question}
        \nThe Response:
        \n{output}

        \nFormatting Rules are as follows:
        \n- Disgard every thing aside from the answer, this includes all thinking processes or justifications for the answer.\n 
        \n- Ensure that the final response contains ONLY lines in this EXACT format: <category> : <information>.\n
        \n 
        \nExample of Acceptable Outputs:
        \nDERMAL : Sensitization
        \nPURITY : 93.4%
        \nNUM SUBJECTS : 45
        \nNull: Null (for non-applicable queries to the study)
        \nDILUTIONS: 10% w/w, 15% w/w, 20% w/w
        \nNot applicable. (acceptable response if the query is not applicable to the study. An alternative answer would be Null:Null)
        \n...etc.
        

        \nYOU MAY START NOW. ADHERE TO THE FORMATTING RULES. Your response should NOT exceed one line. YOU MUST OUTPUT AN ANSWER.
        """
    corrected_output = llm.invoke(template)
    conversation_history = [
        HumanMessage(content=state['final_input']),
        AIMessage(content=output),
        corrected_output
    ]

    return {
        'corrected_output':corrected_output,
        'messages':conversation_history
    }
