from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from typing import Optional

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


class PDFRetrievalChain:
    def __init__(self,persist_directory:Optional[str]=None):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        self.db = None
        self.persist_directory = persist_directory

    def load_docs(self,file):        
        doc = []
        for f in file:
            loader = PDFPlumberLoader(f)
            doc.extend(loader.load())
        return doc
    
    def split_embed(self,doc):
        text_splitter = SemanticChunker(self.embedding_model)
        
        chunks = text_splitter.split_documents(doc)
        if self.persist_directory:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.db = Chroma.from_documents(
                documents = chunks,
                embedding = self.embedding_model,
                persist_directory = self.persist_directory)
            return self.db

        self.db = Chroma.from_documents(
            documents = chunks,
            embedding = self.embedding_model)
        return self.db

    def initialize(self,files):
        docs = self.load_docs(files)
        return self.split_embed(docs)
    
    def search(self,query):
        ret = self.db.as_retriever()
        results = ret.get_relevant_documents(query)
        return results