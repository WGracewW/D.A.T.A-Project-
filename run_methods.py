# Code for the different alogrithm run types
# Last updated: Jan 24, 2026
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from utils_old import search_and_parse, just_parse, calculator_add, calculator, max_min,cleanup_doc,empty_cache,load_saved_embeds,save_vector_store
import time
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import os
import pymupdf
import chromadb
from langchain_chroma import Chroma
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from fpdf import FPDF
from tqdm import tqdm
import re
import pandas as pd
import sys
import gc 

def gen_run(inputs, intro, few_shots, handbook_dir, pdf_dir, chats_dir, vectordb_dir, cache_dir, vector_save_fp, run_number, graph):
    """
    inputs: list. [ question _ , ["query", [keywords] ] ] ... ] where [keywords] = [ ['XOR','Literal','genotoxicity','rat','rats'] , ... ]
    intro: string. introductory line to append to every input.
    few_shots: string. fewshot examples to append to the end of every input, before the context.
    handbook_dir: string. filepath for the student handbook. (stored as vectore store information!)
    pdf_dir: string. folder directory for the studies (pdfs)
    chats_dir: string. folder directory to store the outputs.
    vectordb_dir: string. folderpath for the vector stores.
    cache_dir: string. folderpath for the cache folder.
    vector_save_fp: string. filepath for all the saved vector stores.
    run_number: int. The run number.
    graph: object. The algorithm.
    """

    all_studies = os.listdir(pdf_dir)
    for idx_s, study in enumerate(all_studies):
        study_fp = os.path.join(pdf_dir,study)
        run_store_name = f"{study}_run_{str(run_number)}_full_convo.txt"
        response_store_name = f"{study}_run_{str(run_number)}_response_only.txt"
        start_time_s = time.time()

        if run_store_name in os.listdir(chats_dir): # reponse already stored.
            print(f"Response already recorded for {run_store_name}. Skipping.")
            continue 
        
        else:
            with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as init_r:
                init_r.write(f"Complete Message History for PDF [{study}]\n")
                init_r.write("===========================================================\n")
            with open(os.path.join(chats_dir,response_store_name),'a',encoding ='utf-8') as init:
                init.write(f"Responses Only for PDF [{study}]\n")
                init.write("===========================================================\n")
            for idx,q in enumerate(inputs):
                question_idx = q[0]
                question = q[1][0]
                keywords = q[1][1]
                start_time = time.time()
                response = graph.invoke({
                    'intro':intro,
                    'few_shots':few_shots,
                    'guidebook_fp':handbook_dir,
                    'guide':None,
                    'question':question,
                    'augmented_question':None,
                    'keywords':keywords,
                    'context':[],
                    'tool_result':None,
                    'output':None,
                    'messages':[],
                    'tool_message':None,
                    'pdf_fp':study_fp,
                    'vectordb_dir':vectordb_dir,
                    'cache_dir':cache_dir,
                    'vector_save_fp':vector_save_fp,
                    'vector_store_infos':[],
                    'corrected_output':None
                })
                gc.collect()
                with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                    records = [message.pretty_repr() for message in response['messages']]
                    for r in records:
                        run.write(r + '\n')
                    run.write("===========================================================\n")

                with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                    resp.write(f"\n Question: {question} \n")
                    resp.write(response['corrected_output'].pretty_repr())
                    resp.write("===========================================================\n")
                
                end_time = time.time()
                duration = (end_time-start_time)/60
                print(f'Question {idx+1} of Study {idx_s+1} complete, time took: {duration:.2f} minutes.')
                try:
                    llm.client.close()
                    llm_tool.client.close()
                except Exception as e:
                    pass
            end_time_s = time.time()
            duration_s = (end_time_s - start_time_s)/60
            print(f"Study {idx_s+1} complete.\n timme took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")
            with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as times:
                times.write(f"Study {idx_s+1} complete.\n timme took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")

def flow_run(flowchart_fp, intro, few_shots, handbook_dir, pdf_dir, chats_dir, vectordb_dir, cache_dir, vector_save_fp, run_number, graph, debugging = True):
    """
    flowchart_fp: string. filepath for the flowchart (excel)
    intro: string. introductory line to append to every input.
    few_shots: string. fewshot examples to append to the end of every input, before the context.
    handbook_dir: string. filepath for the student handbook. (stored as vectore store information!)
    pdf_dir: string. folderpath for all the studies (pdfs)
    chats_dir: string. folderpath to store the outputs
    vectordb_dir: string. folderpath for the vector stores.
    cache_dir: string. folderpath for the cache folder.
    vector_save_fp: string. filepath for all the saved vector stores.
    run_number: int. The run number.
    graph: object. Algorithm.
    """

    all_studies = os.listdir(pdf_dir)
    # Start flowchart
    flowchart = pd.read_excel(flowchart_fp)

    for idx_s, study in enumerate(all_studies):
        # Configure settings 
        study_fp = os.path.join(pdf_dir,study)
        run_store_name = f"{study}_run_{str(run_number)}_full_convo.txt"
        response_store_name = f"{study}_run_{str(run_number)}_response_only.txt"

        if os.path.exists(run_store_name):
            print(f"Study {study} has a recorded response. Skipping...")
            continue 
    
        start_time_s = time.time()

        with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as init_r:
            init_r.write(f"Complete Message History for PDF [{study}]")

        with open(os.path.join(chats_dir,response_store_name),'a',encoding ='utf-8') as init:
            init.write(f"Responses Only for PDF [{study}]")
        
        questions = {} # qid : question
        for row in range(1,len(flowchart)): # exclude the first question
            qid = []
            for col in range(len(flowchart.columns)-1):
                qid.append( int(flowchart.iloc[row,col]) )
            qid_str = ",".join(map(str, qid))
            questions[qid_str] = flowchart.iloc[row,8]

        first_prompt = flowchart.iloc[0,8]
        first_answer = graph.invoke({
                    'intro':intro,
                    'few_shots':few_shots,
                    'guidebook_fp':handbook_dir,
                    'guide':None,
                    'question':first_prompt,
                    'augmented_question':None,
                    'keywords':[],
                    'context':[],
                    'tool_result':None,
                    'output':None,
                    'messages':[],
                    'tool_message':None,
                    'pdf_fp':study_fp,
                    'vectordb_dir':vectordb_dir,
                    'cache_dir':cache_dir,
                    'vector_save_fp':vector_save_fp,
                    'vector_store_infos':[],
                    'corrected_output':None
                    })

        study_info = []
        study_info.append(f"Question: {first_prompt}, Answer: {first_answer['corrected_output'].content}")

        if debugging == True:
            print(f"First prompt: {first_prompt}, First response: {first_answer['corrected_output'].pretty_repr()}. \n ----------------------------")
        
        # store responses
        with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                records = [message.pretty_repr() for message in first_answer['messages']]
                for r in records:
                    run.write(r + '\n')

        with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
            resp.write(f"\n\n=======================================\nQuestion: {first_prompt}\n=======================================\n\n")
            resp.write(first_answer['corrected_output'].pretty_repr())

        if "in vivo" in str(first_answer['corrected_output'].content).lower(): # For in vivo studies
            # convert keys to lists of lists
            question_keys = [q.split(",") for q in list(questions.keys())]

            # convert to ints
            for a in question_keys:
                aa = a.copy()
                for i in range(len(aa)):
                    a[i] = int(aa[i])

            # keys are lists of ints
            next_ids = [ n for n in list(question_keys) if ( (n[0] == 1) and (n.count(1) == 1) ) ] # questions pertaining to all in vivo studies only (not complex question)
            complex_next_ids = [ c for c in list(question_keys) if ( (c[0] == 1) and (c.count(0) > 1) ) ] # Questions with follow ups or pertaining to more than one in vivo cases
            gen_qids = [ k for k in list(question_keys) if (k[0] == 2) ] # General questions applicable to all studies
            
            if debugging == True:
                print(f"Asking vivo questions.... Number of general in vivo questions: {len(next_ids)}.... Question ids: {next_ids}. \n ------------------------")
                print(f"Number of complex in vivo questions: {len(complex_next_ids)}... Question ids: {complex_next_ids}. \n--------------------------------")
                print(f"Number of general questions: {len(gen_qids)} ... Question ids: {gen_qids}. \n----------------------------------")

            counter = 0

            for idd in next_ids: # First ask the general in vivo questions
                counter += 1
                idd_str = ",".join(map(str, idd))
                q = questions[idd_str]

                if debugging == True:
                    print(f"Asking general in vivo questions... question {counter} of {len(next_ids)}. \n------------------------------------")

                info = study_info[-1] # Only include the last response
                template = f"""
                            {q} \n 
                            What has already been known about this study: 
                            {info}
                            """

                answer = graph.invoke({
                    'intro':intro,
                    'few_shots':few_shots,
                    'guidebook_fp':handbook_dir,
                    'guide':None,
                    'question':template,
                    'augmented_question':None,
                    'keywords':[],
                    'context':[],
                    'tool_result':None,
                    'output':None,
                    'messages':[],
                    'tool_message':None,
                    'pdf_fp':study_fp,
                    'vectordb_dir':vectordb_dir,
                    'cache_dir':cache_dir,
                    'vector_save_fp':vector_save_fp,
                    'vector_store_infos':[],
                    'corrected_output':None
                    })

                study_info.append(f"Question: {q}, Answer: {str(answer['corrected_output'].content)}")
                
                if debugging == True:
                    print(f"Successfully asked question {counter} of {len(next_ids)}. \n----------------------------------")
            
                # store responses ====================================================================================
                with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                        records = [message.pretty_repr() for message in answer['messages']]
                        for r in records:
                            run.write(r + '\n')

                with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                    resp.write(f"\n\n=======================================\nQuestion: {q}\n=======================================\n\n")
                    resp.write(answer['corrected_output'].pretty_repr())
                # store responses ====================================================================================
                # Cleanup
                gc.collect()

            ccount = 0
            for i in complex_next_ids: # Then ask the complex in vivo questions

                if i[-1] == 0: # This is the leading question
                    ccount += 1
                    
                    i_str = ",".join(map(str, i))

                    q1 = questions[i_str]

                    if debugging == True:
                        print(f"Asking complex in vivo questions. Question {ccount} of {len(complex_next_ids)}. Question: {q1} \n------------------------------------")
                    
                    info = study_info[-1]

                    template = f"""
                            {q1} \n 
                            What has already been known about this study: 
                            {info}
                            """
                    
                    answer = graph.invoke({
                    'intro':intro,
                    'few_shots':few_shots,
                    'guidebook_fp':handbook_dir,
                    'guide':None,
                    'question':template,
                    'augmented_question':None,
                    'keywords':[],
                    'context':[],
                    'tool_result':None,
                    'output':None,
                    'messages':[],
                    'tool_message':None,
                    'pdf_fp':study_fp,
                    'vectordb_dir':vectordb_dir,
                    'cache_dir':cache_dir,
                    'vector_save_fp':vector_save_fp,
                    'vector_store_infos':[],
                    'corrected_output':None
                    })
                    study_info.append(f"Question: {q1}, Answer: {str(answer['corrected_output'].content)}")

                    if debugging == True:
                        print(f"successfully asked question {ccount} of {len(complex_next_ids)}. \n----------------------------")
                    
                    # store responses ====================================================================================
                    with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                            records = [message.pretty_repr() for message in answer['messages']]
                            for r in records:
                                run.write(r + '\n')

                    with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                        resp.write(f"\n\n=======================================\nQuestion: {q1}\n=======================================\n\n")
                        resp.write(answer['corrected_output'].pretty_repr())
                    # store responses ====================================================================================

                    if "true" in str(answer['corrected_output'].content).lower(): # Ask follow up question
                        followup_id = i.copy()
                        followup_id[-1] = 1
                        
                        if followup_id in complex_next_ids:
                            
                            followup_id_str = ",".join(map(str, followup_id))
                            q2 = questions[followup_id_str]
                            info = study_info[-1]

                            if debugging == True:
                                print(f"Question {ccount} is being asked a follow up question: {q2}. \n-----------------------------------")

                            template = f"""
                                {q2} \n 
                                What has already been known about this study: 
                                {info}
                                """
                            
                            answer = graph.invoke({
                            'intro':intro,
                            'few_shots':few_shots,
                            'guidebook_fp':handbook_dir,
                            'guide':None,
                            'question':template,
                            'augmented_question':None,
                            'keywords':[],
                            'context':[],
                            'tool_result':None,
                            'output':None,
                            'messages':[],
                            'tool_message':None,
                            'pdf_fp':study_fp,
                            'vectordb_dir':vectordb_dir,
                            'cache_dir':cache_dir,
                            'vector_save_fp':vector_save_fp,
                            'vector_store_infos':[],
                            'corrected_output':None
                            })
                        
                            study_info.append(f"Question: {q2}, Answer: {str(answer['corrected_output'].content)}")

                            if debugging == True:
                                print(f"Successfully asked follow up question for question {ccount}. \n-----------------------------------")
                            
                            # store responses ====================================================================================
                            with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                                    records = [message.pretty_repr() for message in answer['messages']]
                                    for r in records:
                                        run.write(r + '\n')

                            with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                                resp.write(f"\n\n=======================================\nQuestion: {q2}\n=======================================\n\n")
                                resp.write(answer['corrected_output'].pretty_repr())
                            # store responses ====================================================================================

                        else:
                            print(f"No follow up question found for question: {q1} although 'true' was found in answer, skipping ... \n---------------------------")
                    
                    # Cleanup 
                    gc.collect()

            gcount = 0
            for gid in gen_qids: # finally, ask the general questions
                gcount += 1

                gid_str = ",".join(map(str, gid))
                qg = questions[gid_str]

                if debugging == True:
                    print(f"Asking General questions... Question {gcount} of {len(gen_qids)}. Question: {qg}. \n-----------------------------")

                info = study_info[-1]

                template = f"""
                    {qg} \n 
                    What has already been known about this study: 
                    {info}
                    """

                answer = graph.invoke({
                        'intro':intro,
                        'few_shots':few_shots,
                        'guidebook_fp':handbook_dir,
                        'guide':None,
                        'question':template,
                        'augmented_question':None,
                        'keywords':[],
                        'context':[],
                        'tool_result':None,
                        'output':None,
                        'messages':[],
                        'tool_message':None,
                        'pdf_fp':study_fp,
                        'vectordb_dir':vectordb_dir,
                        'cache_dir':cache_dir,
                        'vector_save_fp':vector_save_fp,
                        'vector_store_infos':[],
                        'corrected_output':None
                        })
                    
                study_info.append(f"Question: {qg}, Answer: {str(answer['corrected_output'].content)}")

                if debugging == True:
                    print(f"Successfully asked general question {gcount}. \n------------------------------------------")
                
                # store responses ====================================================================================
                with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                        records = [message.pretty_repr() for message in answer['messages']]
                        for r in records:
                            run.write(r + '\n')

                with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                    resp.write(f"\n\n=======================================\nQuestion: {qg}\n=======================================\n\n")
                    resp.write(answer['corrected_output'].pretty_repr())
                # store responses ====================================================================================
                # Cleanup 
                gc.collect()

        elif "in vitro" in str(first_answer).lower(): # Not an in vivo study

            # convert keys to lists of lists
            question_keys = [q.split(",") for q in list(questions.keys())]
            for a in question_keys:
                aa = a.copy()
                for i in range(len(aa)):
                    a[i] = int(aa[i])

            next_ids = [n for n in list(question_keys) if (n[0] == 0) ] # non-invivo questions
            gen_qids = [g for g in list(question_keys) if (g[0]==2) ] # general questions 

            if debugging == True:
                print(f"Asking vitro questions.... Number of general in vitro questions: {len(next_ids)}.... Question ids: {next_ids}. \n ------------------------")
                print(f"Number of general questions: {len(gen_qids)} ... Question ids: {gen_qids}. \n----------------------------------")
    
            counter = 0
            for idd in next_ids: # First ask the general in vivo questions
                counter += 1
                idd_str = ",".join(map(str, idd))
                q = questions[idd_str]

                if debugging == True:
                    print(f"Asking general in vivo questions... question {counter} of {len(next_ids)}. \n------------------------------------")

                info = study_info[-1]
                template = f"""
                            {q} \n 
                            What has already been known about this study: 
                            {info}
                            """

                answer = graph.invoke({
                    'intro':intro,
                    'few_shots':few_shots,
                    'guidebook_fp':handbook_dir,
                    'guide':None,
                    'question':template,
                    'augmented_question':None,
                    'keywords':[],
                    'context':[],
                    'tool_result':None,
                    'output':None,
                    'messages':[],
                    'tool_message':None,
                    'pdf_fp':study_fp,
                    'vectordb_dir':vectordb_dir,
                    'cache_dir':cache_dir,
                    'vector_save_fp':vector_save_fp,
                    'vector_store_infos':[],
                    'corrected_output':None
                    })

                study_info.append(f"Question: {q}, Answer: {str(answer['corrected_output'].content)}")
                
                if debugging == True:
                    print(f"Successfully asked question {counter} of {len(next_ids)}. \n----------------------------------")
            
                # store responses ====================================================================================
                with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                        records = [message.pretty_repr() for message in answer['messages']]
                        for r in records:
                            run.write(r + '\n')

                with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                    resp.write(f"\n\n=======================================\nQuestion: {q}\n=======================================\n\n")
                    resp.write(answer['corrected_output'].pretty_repr())
                # store responses ====================================================================================
                # Cleanup
                gc.collect()

            gcount = 0
            for gid in gen_qids: # finally, ask the general questions
                gcount += 1

                gid_str = ",".join(map(str, gid))
                qg = questions[gid_str]

                if debugging == True:
                    print(f"Asking General questions... Question {gcount} of {len(gen_qids)}. Question: {qg}. \n-----------------------------")

                info = study_info[-1]

                template = f"""
                    {qg} \n 
                    What has already been known about this study: 
                    {info}
                    """

                answer = graph.invoke({
                        'intro':intro,
                        'few_shots':few_shots,
                        'guidebook_fp':handbook_dir,
                        'guide':None,
                        'question':template,
                        'augmented_question':None,
                        'keywords':[],
                        'context':[],
                        'tool_result':None,
                        'output':None,
                        'messages':[],
                        'tool_message':None,
                        'pdf_fp':study_fp,
                        'vectordb_dir':vectordb_dir,
                        'cache_dir':cache_dir,
                        'vector_save_fp':vector_save_fp,
                        'vector_store_infos':[],
                        'corrected_output':None
                        })
                    
                study_info.append(f"Question: {qg}, Answer: {str(answer['corrected_output'].content)}")

                if debugging == True:
                    print(f"Successfully asked general question {gcount}. \n------------------------------------------")
                
                # store responses ====================================================================================
                with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                        records = [message.pretty_repr() for message in answer['messages']]
                        for r in records:
                            run.write(r + '\n')

                with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                    resp.write(f"\n\n=======================================\nQuestion: {qg}\n=======================================\n\n")
                    resp.write(answer['corrected_output'].pretty_repr())
                # store responses ====================================================================================
                # Cleanup
                gc.collect()

        else:
            raise ValueError(f"When asked if study is in vitro or in vivo, response could not be interrperted! Response: {first_answer}.")

        # Finally, record the time. 
        end_time_s = time.time()
        duration_s = (end_time_s - start_time_s)/60
        print(f"Study {idx_s+1} complete.\n time took:{duration_s:.2f} minutes. Questions asked: {len(study_info)}")

        with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as times:
            times.write(f"\n Time took:{duration_s:.2f} minutes. Questions asked: {len(study_info)}.")
        
def debug_retrieval_run_9(pdf_dir, chats_dir,debug_store_name,graph):
    all_studies = os.listdir(pdf_dir)

    for idx_s, study in enumerate(all_studies):
        study_fp = os.path.join(pdf_dir,study)
        start_time_s = time.time()
        debug_store_name = f"{study}_debug.txt"

        with open(os.path.join(chats_dir,debug_store_name),'a',encoding='utf-8') as init_r:
            init_r.write("Question idx,Page_Length, Page_Numbers,Time Took")

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
                'messages':[],
                'pdf_fp':study_fp,
                'corrected_output':None,
                'final_input':None,
                'retrieved_pages':None
            })
            gc.collect()
            end_time = time.time()
            duration = (end_time-start_time)/60

            with open(os.path.join(chats_dir,debug_store_name),'a',encoding='utf-8') as run:
                retrieved_pages = response['retrieved_pages']
                run.write("\n")
                run.write(f"{question_idx},{retrieved_pages["length"]},{retrieved_pages["page numbers"]},{duration:.2f}")

            print(f'Question {idx+1} of Study {idx_s+1} complete, time took: {duration:.2f} minutes.')
            try:
                llm.client.close()
                llm_tool.client.close()
            except Exception as e:
                pass
        end_time_s = time.time()
        duration_s = (end_time_s - start_time_s)/60
        print(f"Study {idx_s+1} complete.\n timme took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")

def gen_run_9(inputs, intro, few_shots, handbook_dir, pdf_dir, chats_dir, run_number, debugging, graph):
    """
    inputs: list. [ question _ , ["query", [keywords] ] ] ... ] where [keywords] = [ ['XOR','Literal','genotoxicity','rat','rats'] , ... ]
    intro: string. introductory line to append to every input.
    few_shots: string. fewshot examples to append to the end of every input, before the context.
    handbook_dir: string. filepath for the student handbook. (stored as vectore store information!)
    pdf_dir: string. folder directory for the studies (pdfs)
    chats_dir: string. folder directory to store the outputs.
    run_number: int. The run number.
    debugging: Boolean.
    graph: object. The algorithm.
    """

    all_studies = os.listdir(pdf_dir)
    for idx_s, study in enumerate(all_studies):
        study_fp = os.path.join(pdf_dir,study)
        run_store_name = f"{study}_run_{str(run_number)}_full_convo.txt"
        response_store_name = f"{study}_run_{str(run_number)}_response_only.txt"
        start_time_s = time.time()

        if run_store_name in os.listdir(chats_dir): # reponse already stored.
            print(f"Response already recorded for {run_store_name}. Skipping.")
            continue 
        
        else:
            with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as init_r:
                init_r.write(f"Complete Message History for PDF [{study}]")

            with open(os.path.join(chats_dir,response_store_name),'a',encoding ='utf-8') as init:
                init.write(f"Responses Only for PDF [{study}]")

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
                    'debugging':debugging
                })
                gc.collect()
                with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
                    records = [message.pretty_repr() for message in response['messages']]
                    for r in records:
                        run.write(f"\n{r}")
                    run.write("\n"+"="*80)

                with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
                    resp.write(f"\nQuestion: {question}")
                    resp.write(f"\n{response['corrected_output'].pretty_repr()}")
                    resp.write("\n"+"="*80)
                
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
            with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as times:
                times.write(f"\nStudy {idx_s+1} complete.\n Time took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")