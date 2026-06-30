# Initialization - Run once
# Last updated: Jan 24, 2026
import gc 
def store_guide_as_vector_store(guide_pdf_fp,target_fp,save_info_fp):
    import pymupdf
    import chromadb
    from langchain_chroma import Chroma
    import uuid
    from langchain_huggingface import HuggingFaceEmbeddings
    #from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    import os 
    from utils import save_vector_store

    pdf_name = os.path.basename(guide_pdf_fp).split('.')
    pdf_name.remove('pdf')
    pdf_name = "".join(pdf_name)

    with pymupdf.open(guide_pdf_fp) as doc:
        print("Document Opened.")
        pages = [page.get_text() for page in doc]

        vector_store_infos = None # List[persist_directory,collection_name,ids]

        persist_directory = os.path.join(target_fp,pdf_name)

        # Initialize emebedding
        # need to download embedding function bc for some proxy error for some reason.
        embedding_func = HuggingFaceEmbeddings(model_name=r".\embeddings_local\all-MiniLM-L6-v2")

        collection_name = f"pdf_name_vectorstore_collection_name"
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_func,
            persist_directory=persist_directory # Creates folder to store vector store
        )

        documents = [Document(
                        page_content=text,
                        metadata={},
                        id=str(index)
                    ) for index,text in enumerate(pages)]

        ids = [str(index) for index,text in enumerate(pages)]

        vector_store.add_documents(documents=documents, ids=ids)
        print(f"Successfully created vector store for {pdf_name}. Number of pages: {len(pages)}")

        save_vector_store(persist_directory,collection_name,ids,save_info_fp)

    gc.collect()
        
store_guide_as_vector_store(r"C:\Users\Grace\Documents\Code\DATA_Summer_2025\dependants\Structured EAU1 _student_ handbook (1).pdf",r"C:\Users\Grace\Documents\Code\DATA_Summer_2025\dependants",r"C:\Users\Grace\Documents\Code\DATA_Summer_2025\dependants\guidebook_vsinfo.txt")