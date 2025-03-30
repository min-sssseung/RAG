from rag.pdf_rag import PDFRetrievalChain
import glob
import os

def main():
    path = 'pdf_file/'
    pdf_files = glob.glob(os.path.join(path,'**','*.pdf'),recursive = True)
    print(pdf_files)
    chain = PDFRetrievalChain(
        persist_directory='chroma_db'
    )

    ret = chain.initialize(pdf_files)
    query = input(f"질문 : ")
    results = chain.search(query)
    
    print(results)


if __name__ == "__main__":
    main()