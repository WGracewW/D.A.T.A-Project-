# only for testing purposes
#from langchain_community.llms import LlamaCpp
#from langgraph.graph import StateGraph, START, END
#from langchain_core.messages import AIMessage, HumanMessage
#from langgraph.prebuilt import tools_condition, ToolNode
#from typing_extensions import TypedDict, List, Literal
#from langchain_core.messages import AnyMessage
#from langgraph.graph.message import add_messages
#from langchain_core.documents import Document
#from utils import detect_sections, cleanup_text, is_toc, clean_pymupdf_text
#from run_methods import flow_run, gen_run 
import time
from typing import Annotated
#from langchain_core.prompts import PromptTemplate
import os
import pymupdf
#from langchain_chroma import Chroma
#import chromadb
import uuid
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_ollama.chat_models import ChatOllama
#from fpdf import FPDF
#from tqdm import tqdm
import re
import pandas as pd
import sys 
import unicodedata

def turn_file_into_csv(filepath, out_fp):
    questions = []
    retrieved_contexts = []
    handbook_titles = []

    with open(filepath,'r') as doc:
        for line in doc:
            context = line.strip()
            if "Retrieved context pages" in context:
                match = re.search(r"\[(.*?)\]", context)
                if match:
                    result = match.group(1)
                    retrieved_contexts.append(result)
            
            elif "Question" in context:
                match = re.search(rf"{re.escape('Question')}\s*:\s*(.*)", context)
                questions.append(match.group(1))
            
            elif "Final Retrieved Title" in context:
                match = re.search(rf"{re.escape('Final Retrieved Title')}\s*:\s*(.*)", context)
                if match:
                    handbook_titles.append(match.group(1))
    
    import pandas as pd

    Dataframe = pd.DataFrame({
        "Question":questions,
        "Retrieved Context Pages":retrieved_contexts,
        "Final Handbook Title":handbook_titles
    })
    Dataframe.to_csv(out_fp, index=False)

def evaluate_retrieval_results(results_fp,out_fp):
    results = pd.read_excel(results_fp,dtype=str)
    scores = {
        "Correct Context Retrieval":[],
        "Correct Handbook Retrieval":[],
        "Correct Summary Retrieval":[]
    }
    
    def normalize(s):
        if pd.isna(s):
            return None
        s = str(s)

        # normalize unicode (fixes hidden format differences)
        s = unicodedata.normalize('NFKC', s)

        # remove zero-width characters
        s = re.sub(r'[\u200b-\u200d\ufeff]', '', s)

        # replace non-breaking spaces
        s = s.replace('\xa0', ' ')

        # collapse whitespace
        s = re.sub(r'\s+', ' ', s)

        return s.strip().lower()

    # evaluate context
    for cidx in range(len(results['Correct Context'])):
        cc = results['Correct Context'][cidx]
        retrieved_contexts = results['Context pages'][cidx]

        if pd.isnull(cc): # no correct answer (NA)
            scores["Correct Context Retrieval"].append(1)
            if pd.isnull(retrieved_contexts): # Answer is also NA
                # Check handbook
                handbook_title = results['handbook title'][cidx]
                correct_handbook = results['Correct Handbook Title'][cidx]
                if (normalize(handbook_title) == normalize(correct_handbook)):
                    scores["Correct Handbook Retrieval"].append(1)
                else:
                    scores["Correct Handbook Retrieval"].append(0)

                # Check summary
                if pd.isnull(results['Summary Page'][cidx]):
                    scores['Correct Summary Retrieval'].append(1)
                else:
                    scores['Correct Summary Retrieval'].append(0)
                continue

            retrieved_pages_raw = retrieved_contexts.strip().split(',')
            retrieved_pages = []
            for r in retrieved_pages_raw:
                r = r.strip()
                try:
                    newr = int(r)
                    retrieved_pages.append(newr)
                except:
                    print(f"Retrieved page not interger:{r}.")
                    newr = r
                    retrieved_pages.append(newr)

            # Check handbook
            handbook_title = results['handbook title'][cidx]
            correct_handbook = results['Correct Handbook Title'][cidx]
            if (normalize(handbook_title) == normalize(correct_handbook)):
                scores["Correct Handbook Retrieval"].append(1)
            else:
                scores["Correct Handbook Retrieval"].append(0)
            
            # Check Summary
            summary = results['Summary Page'][cidx]
            if pd.isnull(summary): # No summary page
                scores['Correct Summary Retrieval'].append(1)
                continue

            # Clean up data format
            if '+' in summary:
                summary_page_raw = summary.split('+')
                summary_page = []
                for s in summary_page_raw:
                    s = s.strip()
                    try:
                        news=int(s)
                        summary_page.append(news)
                    except:
                        raise Exception(f"Summary page {summary} contains non-integer numbers after splitting by '+'!!")
                
                matches = 0
                for s in summary_page:
                    if s in retrieved_pages:
                        matches += 1
                
                if matches == len(summary_page):
                    scores['Correct Summary Retrieval'].append(1)
                else:
                    scores['Correct Summary Retrieval'].append(0)
            
            else:
                summary_page_raw = summary.split(',')
                summary_page = []
                for s in summary_page_raw:
                    s = s.strip()
                    try:
                        news = int(s)
                        summary_page.append(news)
                    except:
                        print(f"summary page not interger:{s}.")
                        news = s
                        summary_page.append(news)

                if any(answer in retrieved_pages for answer in summary_page):
                    scores['Correct Summary Retrieval'].append(1)
                else:
                    scores['Correct Summary Retrieval'].append(0)

            continue
        
        elif pd.isnull(retrieved_contexts): # answer is NA
            scores["Correct Context Retrieval"].append(0)
            # Check handbook
            handbook_title = results['handbook title'][cidx]
            correct_handbook = results['Correct Handbook Title'][cidx]
            if (normalize(handbook_title) == normalize(correct_handbook)):
                scores["Correct Handbook Retrieval"].append(1)
            else:
                scores["Correct Handbook Retrieval"].append(0)

            # Check summary
            if pd.isnull(results['Summary Page'][cidx]):
                scores['Correct Summary Retrieval'].append(1)
            else:
                scores['Correct Summary Retrieval'].append(0)

            continue
        
        elif retrieved_contexts == '1000': # Special case: length of document <5 pages, so all pages returned
            scores["Correct Context Retrieval"].append(1)
            scores['Correct Summary Retrieval'].append(1)

            # Check handbook
            handbook_title = results['handbook title'][cidx]
            correct_handbook = results['Correct Handbook Title'][cidx]
            if (normalize(handbook_title) == normalize(correct_handbook)):
                scores["Correct Handbook Retrieval"].append(1)
            else:
                scores["Correct Handbook Retrieval"].append(0)
            
            continue
        
        # Check context
        correct_pages_raw = cc.strip().split(',')
        retrieved_pages_raw = retrieved_contexts.strip().split(',')
        matched_contexts = 0
        # Clean up data format
        correct_pages = []
        retrieved_pages = []
        for r in retrieved_pages_raw:
            r = r.strip()
            try:
                newr = int(r)
                retrieved_pages.append(newr)
            except:
                print(f"Retrieved page not interger:{r}.")
                newr = r
                retrieved_pages.append(newr)

        for c in correct_pages_raw:
            if '+' in c: # Check these cases individually
                c = c.strip()
                cs = c.split('+')
                cmatches = 0
                for page in cs:
                    try:
                        p = int(page)
                        if p in retrieved_pages:
                            cmatches += 1
                    except:
                        raise Exception(f"Correct context pages is not integer after splitting with '+': {c}.")

                if cmatches == len(cs):
                    matched_contexts +=1
                    continue
                else:
                    pass

            else:
                # Then proceed to clean up normally
                c = c.strip()
                try:
                    newc = int(c)
                    correct_pages.append(newc)
                except:
                    raise Exception(f"Non-integer page detected in correct pages when split by ',': {c}")
                    newc = c
                    correct_pages.append(newc)

        if any(answer in retrieved_pages for answer in correct_pages):
            matched_contexts +=1

        if matched_contexts > 0:
            scores['Correct Context Retrieval'].append(1)
        else:
            scores['Correct Context Retrieval'].append(0)
        # Check handbook
        handbook_title = results['handbook title'][cidx]
        correct_handbook = results['Correct Handbook Title'][cidx]
        if (normalize(handbook_title) == normalize(correct_handbook)):
                scores["Correct Handbook Retrieval"].append(1)
        else:
            scores["Correct Handbook Retrieval"].append(0)
        
        # Check Summary
        summary = results['Summary Page'][cidx]
        if pd.isnull(summary): # No summary page
            scores['Correct Summary Retrieval'].append(1)
            continue

        # Clean up data format
        if '+' in summary:
            summary_page_raw = summary.split('+')
            summary_page = []
            for s in summary_page_raw:
                s = s.strip()
                try:
                    news=int(s)
                    summary_page.append(news)
                except:
                    raise Exception(f"Summary page {summary} contains non-integer numbers after splitting by '+'!!")
            
            matches = 0
            for s in summary_page:
                if s in retrieved_pages:
                    matches += 1
            
            if matches == len(summary_page):
                scores['Correct Summary Retrieval'].append(1)
            else:
                scores['Correct Summary Retrieval'].append(0)
        
        else:
            summary_page_raw = summary.split(',')
            summary_page = []
            for s in summary_page_raw:
                s = s.strip()
                try:
                    news = int(s)
                    summary_page.append(news)
                except:
                    print(f"summary page not interger:{s}.")
                    news = s
                    summary_page.append(news)

            if any(answer in retrieved_pages for answer in summary_page):
                scores['Correct Summary Retrieval'].append(1)
            else:
                scores['Correct Summary Retrieval'].append(0)

    scores_df = pd.DataFrame(scores)
    combined_df = pd.concat((results,scores_df),axis=1)
    combined_df.to_csv(out_fp,index=False)

evaluate_retrieval_results(r"C:\Users\CHWANG\Downloads\v9.1_trial3.xlsx", r"C:\Users\CHWANG\OneDrive - HC-SC PHAC-ASPC\Documents\Code\v9.1_trial3_scored.csv")
sys.exit()

text = "Study code:  11/339-020C\nFinal Report\n\nPage  10  of  66\n\nSUMMARY\n\nEthanedioic acid, hydrate (1:2) was tested in vitro in a Chromosome Aberration Assay  using Chinese hamster V79 lung cells. The test item was formulated in phosphate  buffered saline and it was examined up to the cytotoxic concentrations according to  the OECD guideline recommendations. In the performed independent Chromosome\nAberration Assays using duplicate cultures at least 200 well-spread metaphase cells  (or until a clear positive response was detected) were analysed for each test item  treated, negative (vehicle) and positive control sample.\n\nIn Chromosome Aberration Assay 1, a 3-hour treatment with metabolic activation (in  the presence of S9-mix) and a 3-hour treatment without metabolic activation (in the  absence of S9-mix) were performed. Sampling was performed 20 hours after the  beginning of the treatment in both cases. The examined concentrations of the test item  were 400, 200, 150, 125, 100, 75, 50, 25 and 12.5 µg/mL.\n\nIn Assay 1, no insolubility was detected at the end of the treatment period in the final  treatment medium in any of the examined treatment concentrations. No large changes  in pH and osmolality were detected. Cytotoxicity was observed in this assay at 400,\n200, 150, 125 and 100 µg/mL concentrations without metabolic activation (relative  survival values were 7, 10, 18, 28 and 30 %, respectively) and at 400, 200, 150 and\n125 µg/mL concentrations with metabolic activation (relative survival values were 7,\n6, 24 and 46 %, respectively). Therefore, concentrations of 100, 75, 50 and 25 µg/mL  (a total of four) were selected for evaluation in case of the experiment without  metabolic activation and concentrations of 125, 100 and 50 µg/mL (a total of three)  were selected for evaluation in case of the experiment with metabolic activation. None  of the treatment concentrations with or without metabolic activation caused a  significant increase in the number of cells with structural chromosome, thus this assay  was considered to be negative.\n\nIn Chromosome Aberration Assay 2, a 3-hour treatment with metabolic activation (in  the presence of S9-mix) and a 20-hour treatment without metabolic activation (in the  absence of S9-mix) were performed. Sampling was performed 20 hours after the  beginning of the treatment in both cases. The examined concentrations of the test item  were 400, 200, 150, 125, 100, 75, 50, 25 and 12.5 µg/mL.\n\nIn Assay 2, similarly to the first experiment, no insolubility was detected at the end of  the treatment period in the final treatment medium at any of the examined  concentrations. No large changes in pH and osmolality were detected. Cytotoxicity  was observed at 400, 200, 150, 125, 100 and 75 µg/mL concentrations without  metabolic activation (relative survival values were 4, 2, 7, 14, 12 and 24 %,  respectively) and at 400, 200, 150, 125 and 100 µg/mL concentrations (relative  survival values were 6, 11, 14, 30 and 47%, respectively). Concentrations of 100, 75,\n50 and 25 µg/mL (a total of four) were selected for evaluation in case of the  experiment without metabolic activation; and 100, 75 and 50 µg/mL (a total of three)  were selected for evaluation in case of the experiment with metabolic activation."

pdf_fp = r"C:\Users\Grace\Documents\Code\DATA_Summer_2025\pdf\New_studies_20\1.10 Oral_repd.pdf"

Title_patterns = r"^\s*(?:[A-Z][A-Z0-9&''\-]*\s+){0,3}[A-Z][A-Z0-9&''\-]*\s*(?:\((?:cont(?:\.|d|inued)?|continued|contd?)\))?\s*$" # All cap words followed by opt. (cont.) or the like.
    
TOC_TITLE_RE = re.compile(
    r'(?:(?<=\n)|(?<=\A))'
    r'[ \t]*'
    r'('
        r'(?=.*[A-Za-z])'
        r'(?:[A-Z][a-zA-Z]*|[a-z]{1,4})'
        r'(?:[ \t]+(?:[A-Z][a-zA-Z]*|[a-z]{1,4}))*'
        r'|'
        r'(?=.*[A-Za-z])'
        r'[A-Z0-9][A-Z0-9 \t\-–—]{2,}'
    r')'
    r'[ \t]*'
    r'\n+'
    r'\s*'
    r'(?=[A-Z])',
    re.MULTILINE
)

#with pymupdf.open(pdf_fp) as doc:
#    pages = [p.get_text() for p in doc]
#    p = clean_pymupdf_text(pages[20])
#    print(p)
#    print(is_toc(p))

#sys.exit()
    

sections, debug = detect_sections(
    pdf_fp,
    searching = True,
    target_titles = ['summary','sumnary','abstract']
)
#print(debug)
#sys.exit()
for s in sections:
    print(f"Page: {s.page_num}")
    print(f"Title:{s.title}")