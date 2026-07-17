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
import math
from collections import Counter, defaultdict
import pandas as pd
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
from enum import Enum
import numpy as np
import gc

from utils import clean_pymupdf_text, clean_prompt_input, detect_sections, is_toc, is_table, ocr_docling
from run_methods import gen_run_9_2

# Last Edited: July 16, 2026

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

# Generic, not domain-specific — just enough to keep candidate phrases from
# being pure filler ("the", "of", "is"...). Safe to trim/extend.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "this", "that", "these", "those", "it",
    "its", "does", "do", "did", "has", "have", "had", "can", "could",
    "should", "would", "will", "shall", "may", "might", "not", "so",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b(?:[A-Za-z]+(?:/[A-Za-z]+)?|\d+(?:\.\d+)?%?)\b|%", text.lower()) # All alphanumeric numbers and instances containing '/' and '%' but none of the other symbols (like puncutation marks)

def generate_candidates(text: str, max_n: int = 2) -> list[str]:
    """
    Generic n-gram candidate generator (1 to max_n words), dropping
    candidates that are entirely stopwords or start/end on a stopword
    (keeps 'test guideline' but rejects 'to the' or 'in a').
    """
    tokens = tokenize(text)
    candidates = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i + n]
            if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                continue
            if all(t in STOPWORDS for t in gram):
                continue
            candidates.append(" ".join(gram))
    return list(dict.fromkeys(candidates))  # dedupe, keep order

class SimpleBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = tokenized_corpus
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in tokenized_corpus]
        self.idf: dict[str, float] = self._compute_idf()
        self.max_idf = max(self.idf.values()) if self.idf else 1.0

    def _compute_idf(self) -> dict[str, float]:
        df = defaultdict(int)
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        n = len(self.corpus)
        return {term: math.log((n - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()}

    def get_scores(self, query_tokens: list[str], term_weights: dict[str, float] | None = None) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for term in query_tokens:
            if term not in self.idf:
                continue
            weight = term_weights.get(term, 1.0) if term_weights else 1.0
            term_idf = self.idf[term]
            for i, doc_freqs in enumerate(self.doc_freqs):
                freq = doc_freqs.get(term, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lens[i]
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[i] += weight * term_idf * (freq * (self.k1 + 1)) / denom
        return scores


class VectorStore:
    def __init__(self, model_dir: str = r".\embeddings_local\all-MiniLM-L6-v2", keywords:list[str]| None = None):
        self.model = SentenceTransformer(model_dir)
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: SimpleBM25 | None = None
        self._tokenized_corpus: list[list[str]] = []
        self.keywords = keywords

    def add_documents(self, documents: list[Document]):
        new_embeddings = self.model.encode(
            [doc.content for doc in documents],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.documents.extend(documents)
        self.embeddings = (
            new_embeddings if self.embeddings is None
            else np.vstack([self.embeddings, new_embeddings])
        )
        self._tokenized_corpus.extend(tokenize(doc.content) for doc in documents)
        self.bm25 = SimpleBM25(self._tokenized_corpus)

    def _extract_keywords(self, query: str, query_embedding: np.ndarray, top_frac: float = 0.1):
        """
        Combines two independent signals per candidate phrase:
        - semantic centrality: cosine sim between candidate's embedding and
          the full query's embedding (via your local model)
        - corpus rarity: average IDF of the candidate's tokens, normalized
          by the corpus's max IDF (from BM25, already computed)
        Returns {phrase: combined_score} for the top_frac fraction of candidates.
        """
        candidates = generate_candidates(query)
        if not candidates:
            return {}

        candidate_embeddings = self.model.encode(candidates, normalize_embeddings=True)
        semantic_scores = (candidate_embeddings @ query_embedding.T).flatten()

        idf_scores = []
        for phrase in candidates:
            toks = phrase.split()
            avg_idf = sum(self.bm25.idf.get(t, 0.0) for t in toks) / len(toks)
            idf_scores.append(avg_idf / self.bm25.max_idf if self.bm25.max_idf else 0.0)
        idf_scores = np.array(idf_scores)

        # blend of "central to the query's meaning" and "rare in the corpus"
        combined = 0.7 * semantic_scores + 0.3 * idf_scores

        n_keep = max(1, int(len(candidates) * top_frac))
        top_idx = np.argsort(combined)[::-1][:n_keep]

        return {candidates[i]: float(combined[i]) for i in top_idx}

    def search(
        self,
        query: str,
        k: int = 3,
        score_threshold: float | None = None,
        candidate_pool: int = 15,
        rrf_k: int = 60,
        keyword_boost: float = 2.0,
    ):
        if self.embeddings is None or len(self.documents) == 0:
            raise ValueError("Vectorstore is empty. Call add_documents() first.")

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        cosine_scores = (self.embeddings @ query_embedding.T).flatten()

        # --- keyword-aware BM25 ---
        if (self.keywords is not None) and (len(self.keywords)>=1):
            keywords = {i : 1.0 for i in self.keywords}
        else:
            keywords = self._extract_keywords(query, query_embedding)  # {phrase: score in [0,1]-ish}

        query_tokens = tokenize(query)
        term_weights = {tok: 1.0 for tok in query_tokens}
        for phrase, kw_score in keywords.items():
            for tok in phrase.split():
                boosted = 1.0 + keyword_boost * kw_score
                term_weights[tok] = max(term_weights.get(tok, 1.0), boosted)

        bm25_scores = np.array(self.bm25.get_scores(query_tokens, term_weights=term_weights))

        # flat bonus for verbatim multi-word phrase matches, weighted by the phrase's own score
        phrase_keywords = {p: s for p, s in keywords.items() if " " in p}
        if phrase_keywords:
            for i, doc in enumerate(self.documents):
                content_lower = doc.content.lower()
                for phrase, kw_score in phrase_keywords.items():
                    if phrase in content_lower:
                        bm25_scores[i] += keyword_boost * kw_score

        # --- fuse via RRF ---
        cosine_rank = np.argsort(cosine_scores)[::-1]
        bm25_rank = np.argsort(bm25_scores)[::-1]

        n = min(candidate_pool, len(self.documents))
        cosine_rank_pos = {idx: r for r, idx in enumerate(cosine_rank[:n])}
        bm25_rank_pos = {idx: r for r, idx in enumerate(bm25_rank[:n])}
        candidate_indices = set(cosine_rank_pos) | set(bm25_rank_pos)

        fused_scores = {}
        for idx in candidate_indices:
            score = 0.0
            if idx in cosine_rank_pos:
                score += 1 / (rrf_k + cosine_rank_pos[idx] + 1)
            if idx in bm25_rank_pos:
                score += 1 / (rrf_k + bm25_rank_pos[idx] + 1)
            fused_scores[idx] = score

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = [
            SearchResult(document=self.documents[idx], score=float(score))
            for idx, score in ranked
        ]

        if score_threshold is not None:
            results = [r for r in results if r.score >= score_threshold]

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
    chats_dir:str
    augmented_question:str
    context:List[Document]
    output:str 
    messages: Annotated[list[AnyMessage], add_messages] 
    pdf_fp:str
    corrected_output:str
    final_input:str
    retrieved_pages:dict
    debugging:bool
    keywords:list[str]
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

            return {
                'context':non_toc_pages,
                'retrieved_pages':retrieved_pages
            }
        
        # Else
        for pdx in range(len(pages)):
            p = pages[pdx]
            documents.append(Document(content=p,metadata={"page":pdx+1})) # Maybe add the page title as well here?
    
    # Create VectorStore
    keywords = state['keywords']
    store = VectorStore(model_dir = r".\embeddings_local\all-MiniLM-L6-v2",keywords=keywords)
    store.add_documents(documents=documents)
    rerank_results = store.search(query=query)
    context = []
    page_numbers = []
    # Get rid of TOC pages from retrieval and begin adding to context
    for c in rerank_results:
        page_text = c.document.content
        page_number = c.document.metadata["page"]
        if is_toc(page_text) == False:

            if is_table(page_text):
                # use marker ocr for tables
                text = ocr_docling(state["pdf_fp"],start_page=page_number) # 1-based page numbers
                context.append(text)
                page_numbers.append(page_number)

            else:
                context.append(page_text)
                page_numbers.append(page_number)

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

    return {
        'context':context,
        'retrieved_pages': retrieved_pages
    }

def augment(state:GraphState):

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
# input format example ['question 1',['what is...?' -> query,['micronucles','in vivo' -> keywords]]]
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
        """,['purity']]
    ]
    ,
    ['question 3',[
        """
        Find the vehicle(s) or solvent(s) used in this study to dissolve the test substance.\n
        Look for pages containing text like "w/v", "%", "v/v", they usually tell you the vehicle used for dissolving the test substance.\n
        Examples include: alcohol, water, methanol, DMSO, oils, aqueous methylcellulose, acetone, petrolatum, sodium chloride, gelatin capsule — but other answers are allowed.
        """,[]]]
    ,
    ['question 4',[
        """
        Test guidelines can help legitimize studies. Guidelines are often from OECD, ECC or EC. For example, OECD 471 is a type of guideline. Does this study follow a test guideline?
        """,['OECD','ECC','EC']
    ]],
    ['question 5',[
        """
        Test methods are well known methods that say what kind of study is being performed. Guineau pig maximisation,
        \nAmes test, Micronucleus test, Human Repeat Insult Patch test, Guineapig Maximization test are examples of test methods. Does this study follow a test method?
        """,[]
    ]],
    ['question 6',[
        """
        What was the maximum dosage of the test substance used on the test subject, with unit? Please note that doses can have various units, such as % (percentage), mg/kg, mg/plate ...etc.\nDo NOT report the dose used in the solubility study, only report the highest dose used on the test subject.
        """, []
    ]],
    ['question 7',[
        """
        Is the substance ever diluted?\n Look for symbols like "w/v", "%", "v/v" ...etc.\n If yes, state the dilution percentage or percentages. If no, answer "null". You do NOT need to state the solvent, only the dilution percentages or 'null'.
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
        Find the Hazard Classification of this study.\n 
        Was the substance classified as low, low-moderate, moderate, moderate-high, high, or extreme hazard? If there is no info, answer "null".\n
        If there is NO mention of NOAEL, negative effect level, NOEL, NOEC, or anything of the like in the study, report 'not applicable'.\n
        If the toxicity endpoint of the study was reported to be negative or inconclusive, report 'not applicable'.\n
        """, []
    ]],
    ['question 10',[
        """
        Answer this question only if the study is a *repeated dose* study. Otherwise, answer 'not applicable'. \n 
        What was the TESTING duration of the study (not the dates during which the study is conducted), including units (days, weeks, years). If it's not mentioned, answer "null".\n
        Do NOT explain the methodology or categorize the study into 'Full study' or 'Summary'; ONLY state the duration of the study IF it is a *repeat dose* study.\n
        """, []
    ]],
    ['question 11',[
        """
        Answer this question only if the study is a *repeated dose* study. Otherwise, answer 'not applicable'. \n 
        What Critical Effects on the test animals changed the NOAEL or the classification hazard? This could be changes in food consumption, organ weight change, weight change, or any other health problem observed in the animal due to the substance. If there were none, answer "null".\n
        Do NOT respond with the hazard classification, only respond if there were Critical Effects that CHANGED the OUTCOME of the study.
        """, []
    ]]]

# Initialization --------------------------------------------------------------------------------------------
chats_dir = r"C:\Users\Grace\Documents\Code\DATA_Summer_2025\chats\v9.2_old_31_N"
pdf_dir = r".\pdf\New_studies_31"
handbook_dir = r".\dependants\Structured EAU1 _student_ handbook (2).pdf"
# Set up counter ----------------------------------------------------------------------------------------
time_per_trial = []
time_per_run = [] # for multiple runs (multiple pdfs)

# Set up query feeding ----------------------------------------------------------------------------------
intro = "You are a chemical toxicity evaluator. Your job is to read a toxicity report and retrieve specific information from the report."
few_shots = "Format your answer as: <CATEGORY>: <ANSWER>. \n If either is missing or unclear, return Null. \n Examples of acceptable answers:\n DERMAL: Topical Application  \n ORAL: gavage  \n Null : Null \n PURITY: 92% \n MAX DOSE: 50% w/w \n ...etc."
run_number = 1

if __name__ == "__main__":
    gen_run_9_2(inputs, intro, few_shots, handbook_dir, pdf_dir, chats_dir, run_number, debugging=True, graph=graph)