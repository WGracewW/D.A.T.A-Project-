from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import time
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import os
import pymupdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from tqdm import tqdm
import re
import pandas as pd
import sys
from sentence_transformers import SentenceTransformer
from enum import Enum
import numpy as np
import gc

from utils import clean_pymupdf_text, clean_prompt_input, detect_sections, is_toc
from run_methods import gen_run_9

# Last Edited: June 12, 2026

# Define Classes for Semantic Search --------------------------------------------------

class SimilarityMetric(Enum):
    COSINE='cosine'
    L2='l2'

class Document:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

class SearchResult:
    def __init__(self, document: Document, score: float):
        self.document = document
        self.score = score

    def __repr__(self):
        preview = self.document.content[:80]
        return f"SearchResult(score={self.score:.4f}, content='{preview}...')"

class VectorStore:
    def __init__(self, model_dir: str = r".\embeddings_local\all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_dir)
        self.documents:list[Document] = []
        self.embeddings:np.ndarray | None = None

    def add_documents(self, documents:list[Document]):
        # Embed docs and store them as vectorstore
        new_embeddings = self.model.encode(
            [doc.content for doc in documents],
            normalize_embeddings = True,
            show_progress_bar = True,
        )
        self.documents.extend(documents)
        self.embeddings = (
            new_embeddings
            if self.embeddings is None
            else np.vstack([self.embeddings, new_embeddings])
        )

    def search(self, query:str, score_threshold:float | None = None, k:int=5, metric:SimilarityMetric=SimilarityMetric.COSINE):
        """
        inputs:
        query:str
        k - top-k returns: int
        metric
        score_threshold:float

        output:
        List of SearchResult objects
        """

        if self.embeddings is None or len(self.documents) == 0:
            raise ValueError("Vectorstore is empty. Call add_documents() first.")
        
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        )

        if metric == SimilarityMetric.COSINE:
            # dot product of unit vectors
            scores = (self.embeddings @ query_embedding.T).flatten()
            sorted_indices = np.argsort(scores)[::-1]
            top_indices = sorted_indices[:k]

            results = [
                SearchResult(document=self.documents[i],score=float(scores[i])) for i in top_indices
            ]
            if score_threshold is not None:
                results = [r for r in results if r.score >= score_threshold]
            
        else:
            # L2 simlilarity = sqrt(2 * (1 - cosine_similarity) )
            cosine_scores = (self.embeddings @ query_embedding.T).flatten()
            scores = np.sqrt(np.clip(2 * (1 - cosine_scores), 0, None))
            sorted_indices = np.argsort(scores) # ascending order
            top_indices = sorted_indices[:k]

            results = [
                SearchResult(document = self.documents[i], score=float(scores[i])) for i in scores
            ]

            if score_threshold is not None:
                results = [r for r in results if r.score <= score_threshold]
            
        return results

# Define the local QNA model -------------------------------------------------------------------------------------
llm = ChatOllama(
    model = 'phi4',
    num_ctx = 16000,
    temperature = 0.7,
    verbose = False,
    num_gpu = 1, # number of gpus to use
    num_predict = 500,
    repeat_penalty = 1.2,
    top_k = 50,
    top_p = 0.85
)

# Initialize State Schemas --------------------------------------------------------------------------------------
class GraphState(TypedDict):
    intro: str
    guidebook_fp:str
    guide:str
    question:str
    few_shots:str
    chats_dir:str # debuggin ---
    augmented_question:str
    context:List[Document]
    output:str 
    messages: Annotated[list[AnyMessage], add_messages] 
    pdf_fp:str
    corrected_output:str
    final_input:str
    retrieved_pages:dict
    debugging:bool
# retrieved pages for debugging

# Define Nodes --------------------------------------------------------------------------------------------------------

def retrieve_guide(state:GraphState):
    if state['debugging'] == True:
        print("Retrieving Guide...")
    
    query = state['question']
    guidebook_fp = state['guidebook_fp']

    relevent_page_content = None
    relevent_title = None

    with pymupdf.open(guidebook_fp) as doc:
        pages = [page.get_text() for page in doc]
        titles = []
        for page in pages:
            match_title = re.search(r'\+\+\+([^+]+)\+\+\+', page)

            if match_title:
                titles.append(match_title.group(1).strip())

        q = f"""
            Here are a list of titles from a manual containing information on how to identify the answers to a certain user's query. Pick the section from an evaluator's manual that best describes
            the category of the query. The section should handle and explain how to solve the user's query almost perfectly; if no sections feel right, simply output "Non".\n
            If the query asks for the 'test method', output 'Non'!!\n
            Sections titles:\n{"\n".join(titles)}\n
            User's Query: {query} \n
            You should only choose ONE or NONE title(s) that best describes the query. Output only the best-match Title or 'Non', no other text.\n
            Examples of acceptable outputs: "Vehicle / Solvent", "Test Item Concentration / Dilution" ...etc.\n
            Examples of unacceptable outputs: "The best titles that match the query are 'vehicles' and 'dilution' and 'sample size'" ...etc.\n
            If no titles are tightly relevant to the query, output "Non"! No other text is allowed; you do not need to explain your reasoning. \n
            Remember, If the query asks for the 'test method', output 'Non'.\n
            You may begin now.
            """
        
        a = llm.invoke(q)

        # Count the last occuring matching title; In the event that the model begins to explain its choice, the last matching title should almost certainly be the answer.
        number_of_occurances = {}
        for title in titles:
            if a.pretty_repr().lower().strip().count(title.lower()) >= 1:
                number_of_occurances[title] = a.pretty_repr().count(title)
        
        # Count 'Non' - add 'non' as a title to number_of_occurances if non exists in answer
        non_match = r"\bnon\b"
        if len(re.findall(non_match,a.pretty_repr().lower().strip())) >= 1:
            number_of_occurances['non'] = a.pretty_repr().lower().strip().count('non')
        
        if len(number_of_occurances) > 1: # More than 1 titles appeared in the response
            last_indexes = {}
            for title, value in number_of_occurances.items():
                last_index = a.pretty_repr().rfind(title) # Case Sensitive!
                last_indexes[title] = last_index

            if 'non' in number_of_occurances.keys():
                non_idxs = [ m.start() for m in re.finditer(non_match, a.pretty_repr().lower().strip()) ]
                if len(non_idxs) > 0:
                    last_non_idx = non_idxs[-1]
                    last_indexes['non'] = last_non_idx

            last_title = max(last_indexes, key=last_indexes.get) # last occuring title

            # Check if last occuring title is a "negative case" or not. (i.e. " 'Title' is not applicable for this case.")
            def is_negative(last_title:str, window_front_cut: int, window_end_cut:int, response:str):
                negative_pattern = fr"""
                    (?:["']?{re.escape(last_title)}["']?(?:\s+\w+){{0,2}}\s+\b(?:no|not|wrong)\b)
                    |
                    (?:\bnot\b\s+["']?{re.escape(last_title)}["']?)
                """
                if window_front_cut and window_end_cut:
                    string_to_search = response[window_front_cut:window_end_cut]

                elif (window_front_cut) and not (window_end_cut):
                    string_to_search = response[window_front_cut:]

                else:
                    string_to_search = response

                return bool(re.search(
                        negative_pattern,
                        string_to_search,
                        flags=re.IGNORECASE | re.VERBOSE
                    ))

            # Now check
            filtered_last_indexes = last_indexes.copy()
            is_last_title_negative = is_negative(last_title, filtered_last_indexes[last_title] - 50, None, a.pretty_repr().lower().strip())

            while is_last_title_negative:
                del filtered_last_indexes[last_title]

                if not filtered_last_indexes: # no more items left in list
                    last_title = None
                    break

                last_title = max(filtered_last_indexes, key=filtered_last_indexes.get)

                is_last_title_negative = is_negative( last_title, filtered_last_indexes[last_title] - 50, None, a.pretty_repr().lower().strip() )

            relevent_title = last_title

            #if relevent_title == 'non': # end early if last title is non
            #    return {'guide':None}
            
            for p in pages:
                if f"+++{relevent_title}+++" in p:
                    relevent_page_content = p
        
        else: # only 1 title in response
            relevent_title = next(iter(number_of_occurances),None)
            for p in pages:
                if f"+++{relevent_title}+++" in p:
                    relevent_page_content = p
        
        # for debuggin ---
        print(f"saving handbook results...")
        save_file = "guide_retrieval_debugging_trial3.txt"
        folder = state['chats_dir']
        with open(os.path.join(folder,save_file),'a') as debug:
            debug.write(f"\nRetrieved handbook section (model response): \n{a.pretty_repr()}")
            debug.write(f"\n" + "="*80)
            debug.write(f"\nFinal Retrieved Title: {relevent_title}")
        # for debuggin ---

    return {'guide':relevent_page_content}

def retrieve(state: GraphState):
    if state['debugging'] == True:
        print("Retrieving Context...")
    query = state['question']
    pdf = state['pdf_fp']
    
    documents = []
    retrieved_pages = {} # for debugging!

    # Create document objects
    with pymupdf.open(pdf) as doc:
        print(f"{pdf} Opened.")
        pages_raw = [page.get_text() for page in doc]
        pages = [clean_pymupdf_text(page) for page in pages_raw] # Clean up text

        # ! If total length is <= 5 pages, return all pages (except TOC)
        if len(pages) <= 5:
            non_toc_pages = [page for page in pages if is_toc(page)==False]

            retrieved_pages['length'] = len(non_toc_pages)
            retrieved_pages['page numbers'] = [p+1 for p in range(len(non_toc_pages))]

            # debuggin ---
            save_file = "guide_retrieval_debugging_trial3.txt"
            folder = state['chats_dir']
            with open(os.path.join(folder,save_file),'a') as debug:
                debug.write(f"\nQuestion: {query}")
                debug.write(f"\nRetrieved context pages: {retrieved_pages['page numbers']}")
            # debuggin ---

            return {
                'context':non_toc_pages,
                'retrieved_pages':retrieved_pages
            }
        
        # Else
        for pdx in range(len(pages)):
            p = pages[pdx]
            documents.append(Document(content=p,metadata={"page":pdx+1})) # Maybe add the page title as well here?
    
    # Create VectorStore
    store = VectorStore(model_dir = r".\embeddings_local\all-MiniLM-L6-v2")
    store.add_documents(documents=documents)
    cosine_results = store.search(query, k=2, metric=SimilarityMetric.COSINE)
    context = []
    page_numbers = []
    # Get rid of TOC pages from retrieval and begin adding to context
    for c in cosine_results:
        page_text = c.document.content
        if is_toc(page_text) == False:
            context.append(page_text)
            page_numbers.append(c.document.metadata["page"])

    # Append relevant sections to the context
    Targets = ['summary','sumnary','abstract'] # Includes mis-spellings (wrongful interpretations by OCR)
    target_sections = detect_sections(pdf_fp = state['pdf_fp'], target_titles=Targets, searching=True)
    if len(target_sections)>0:
        for section in target_sections:
            page_number = section.page_num
            if (page_number not in page_numbers): # Page not already included in context.
                context.append(section.content)
                page_numbers.append(page_number)

    retrieved_pages["length"] = int(len(context))
    retrieved_pages["page numbers"] = page_numbers

    # debuggin ---
    print(f"saving context results...")
    save_file = "guide_retrieval_debugging_trial3.txt"
    folder = state['chats_dir']
    with open(os.path.join(folder,save_file),'a') as debug:
        debug.write(f"\nQuestion: {query}")
        debug.write(f"\nRetrieved context pages: {retrieved_pages['page numbers']}")
    
    return {}
    # debuggin ---

    return {
        'context':context,
        'retrieved_pages': retrieved_pages
    }

def augment(state:GraphState):
    # debuggin ---
    return{}
    # debuggin ---

    if state['debugging'] == True:
        print("Augmenting...")
    
    docs = state['context']
    intro = state['intro']
    few_shots = state['few_shots']
    question = state['question']
    guide = state['guide']
    
    if guide is not None:
        if len(state['context']) > 0:
            texts = '\n\n'.join(doc for doc in docs)
            # debugging
            print("Cleaning Pymupdf Text...")
            clean_texts = clean_pymupdf_text(texts)

            input_text = f"""
                {intro}
                \n{question}
                \nYou should use the following guide to retrieve your information:
                \n{guide}
                \n{few_shots}
                \nStudy Report (raw text):
                \nIf a section is irrelevant, nonsensical, or does not help answer the question, ignore it.
                \n--------------------------------------------------------------------------BEGIN EXERPT--------------------------------------------------------------------------
                \n{clean_texts}
                \n--------------------------------------------------------------------------END EXERPT--------------------------------------------------------------------------
                \nYOU MAY NOW WRITE YOUR ANSWER, STOP GENERATING after you've answered the question, you MUST output an answer.
            """
        else:
            input_text = f"""
                Question:\n{state['question']}
                \nIMPORTANT: There is no information found on the toxicology report that may provide an answer to the question. This question has no answer.
                """

    elif guide is None:
        if len(state['context']) > 0:
            texts = '\n\n'.join(doc for doc in docs)
            clean_texts = clean_pymupdf_text(texts)

            input_text = f"""
                \n{intro}
                \n{question}
                \n{few_shots}
                \nStudy Report (raw text):
                \nIf a section is irrelevant, nonsensical, or does not help answer the question, ignore it.
                \n--------------------------------------------------------------------------BEGIN EXERPT--------------------------------------------------------------------------
                \n{clean_texts}
                \n--------------------------------------------------------------------------END EXERPT--------------------------------------------------------------------------
                \nYOU MAY NOW WRITE YOUR ANSWER, STOP GENERATING after you've answered the question, you MUST output an answer.
            """
        else:
            input_text = f"""
                Question:\n{state['question']}
                \nIMPORTANT: There is no information found on the toxicology report that may provide an answer to the question. This question has no answer.
                """

    return{'augmented_question':input_text , 'context':docs}

def generate(state:GraphState):
    # debuggin ---
    return {}
    # debuggin ---

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

def formatter(state:GraphState):
    # debuggin ---
    return {}
    # debuggin ---

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
        

        \nYOU MAY START NOW. ADHERE TO THE FORMATTING RULES. Your response should not exceed one line. YOU MUST OUTPUT AN ANSWER.
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

#-----------------------------------------------------Build Graph ---------------------------------------
builder = StateGraph(GraphState)
# Nodes
builder.add_node("retriever_1",retrieve)
builder.add_node('retrieve_guide_1',retrieve_guide)
builder.add_node('augment_1',augment)
builder.add_node('generate_1',generate)
builder.add_node('formatter',formatter)

#Edges
builder.add_edge(START, 'retriever_1')
builder.add_edge('retriever_1', 'retrieve_guide_1')
builder.add_edge('retrieve_guide_1','augment_1')
builder.add_edge('augment_1', 'generate_1')
builder.add_edge('generate_1', 'formatter')
builder.add_edge('formatter',END)

graph = builder.compile()

#-----------------------------------------------Run Graph ------------------------------------------------
# Set up questions --------------------------------------------------------------------------------------
# input format example ['question 1',['what is...?',['micronucles','in vivo']]]
inputs = [
    ['question 1',[
        """
            Determine the exposure type (ORAL, DERMAL, or INHALATION) for the toxicology study, then classify the exposure method based on these rules:\n
            -DERMAL study exposure methods: Topical Application, Intradermal Injection, or Occlusive Patch \n
            -ORAL study exposure methods: gavage or feed  \n
            -INHALATION study exposure methods: Powder, Vapor, or Gas Chamber\n
            THIS QUESTION DOES NOT APPLY TO IN VITRO STUDIES! (non-applicable)
    """,[]
    ]],
    ['question 2',[
        """
        Find the purity of the tested substance for this toxicology report.
        """,[ ['or','similar','purity'] ]]]
    ,
    ['question 3',[
        """
        Find the vehicle(s) or solvent(s) used in this toxicity study.\n
        Valid examples include: alcohol, water, methanol, DMSO, oils, aqueous methylcellulose, acetone, petrolatum, sodium chloride, gelatin capsule — but other answers are allowed.
        """,[]]
    ],
    ['question 4',[
        """
        Test guidelines can help legitimize studies. Guidelines are often from OECD, ECC or EC. For example, OECD 471 is a type of guideline. Does this study follow a test guideline?
        """,[]
    ]],
    ['question 5',[
        """
        Test methods are well known methods that say what kind of study is being performed. Guineau pig maximisation,
        \nAmes test, Micronucleus test, Human Repeat Insult Patch test, Guineapig Maximization test are examples of test methods. Does this study follow a test method?
        """,[]
    ]],
    ['question 6',[
        """
        What was the maximum dosage of the test substance used on the test subject, with unit? Please note that doses can have various units, such as % (percentage), mg/kg, mg/plate ...etc.
        """, []
    ]],
    ['question 7',[
        """
        Is the substance ever diluted? If yes, state the dilution percentage or percentages. If no, answer "null". You do NOT need to state the solvent, only the dilution percentages or 'null'.
        """, []
    ]],
    ['question 8',[
        """
        What is the total number of animals used in the study? If it is not mentioned, answer "null".
        """, []
    ]],
    ['question 9',[
        """
        Answer this question only if the study is a *repeated dose or sensitization* study, otherwise, answer 'not applicable'.\n 
        Was the substance classified as low, low-moderate, moderate, moderate-high, high, or extreme hazard? If there is no info, answer "null".
        """, []
    ]],
    ['question 10',[
        """
        Answer this question only if the study is a *repeated dose* study. Otherwise, answer 'not applicable'. \n 
        What was the duration of the study, including units (days, weeks, years). If it's not mentioned, answer "null".\n
        Do NOT explain the methodology or categorize the study into 'Full study' or 'Summary'; ONLY state the duration of the study IF it is a *repeat dose* study.\n
        """, []
    ]],
    ['question 11',[
        """
        Answer this question only if the study is a *repeated dose* study. Otherwise, answer 'not applicable'. \n 
        What critical effects on the test animals changed the NOAEL or the classification hazard? This could be changes in food consumption, organ weight change, weight change, or any other health problem observed in the animal due to the substance. If there were none, answer "null".\n
        Do not respond with the hazard classification, only respond if there were critical effects that changed the outcome of the study.
        """, []
    ]]]

# Initialization --------------------------------------------------------------------------------------------
chats_dir = r"C:\Users\Grace\Documents\Code\DATA_Summer_2025\chats\v9.1_old_20_N"
pdf_dir = r".\pdf\New_studies_20"
handbook_dir = r".\dependants\Structured EAU1 _student_ handbook (2).pdf"
# Set up counter ----------------------------------------------------------------------------------------
time_per_trial = []
time_per_run = [] # for multiple runs (multiple pdfs)

# Set up query feeding ----------------------------------------------------------------------------------
intro = "You are a chemical toxicity evaluator. Your job is to read a toxicity report and retrieve specific information from the report."
few_shots = "Format your answer as: <CATEGORY>: <ANSWER>. \n If either is missing or unclear, return Null. \n Examples of acceptable answers:\n DERMAL: Topical Application  \n ORAL: gavage  \n Null : Null \n PURITY: 92% \n MAX DOSE: 50% w/w \n ...etc."
run_number = 1

# debuggin ---
all_studies = os.listdir(pdf_dir)
for idx_s, study in enumerate(all_studies):
    study_fp = os.path.join(pdf_dir,study)
    start_time_s = time.time()
    save_file = "guide_retrieval_debugging_trial2.txt"
    folder = chats_dir
    with open(os.path.join(folder,save_file),'a') as debug:
        debug.write(f"\nStudy Name: {study}")

    for idx,q in enumerate(inputs):
        question_idx = q[0]
        question = q[1][0]
        start_time = time.time()
        response = graph.invoke({
            'intro':intro,
            'few_shots':few_shots,
            'guidebook_fp':handbook_dir,
            'guide':None,
            'question':question,
            'augmented_question':None,
            'context':[],
            'output':None,
            'chats_dir':chats_dir,
            'messages':[],
            'pdf_fp':study_fp,
            'corrected_output':None,
            'retrieved_pages':None,
            'debugging':True
        })
        gc.collect()
        end_time = time.time()
        duration = (end_time-start_time)/60
        print(f'\nQuestion {idx+1} of Study {idx_s+1} complete, Time took: {duration:.2f} minutes.')
        try:
            llm.client.close()
            llm_tool.client.close()
        except Exception as e:
            pass
    end_time_s = time.time()
    duration_s = (end_time_s - start_time_s)/60
    print(f"Study {idx_s+1} complete.\n Time took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")



