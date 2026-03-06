import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import model_config as conf

class VectoreStore:
    def __init__(self, device=None, file_path=None, filter_value=None, filter_type=None, chunk_size=1500, chunk_overlap=150):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.embedding_function = HuggingFaceEmbeddings(model_name=conf.embedding_model_path, model_kwargs={"device": self.device})
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.filter_value = filter_value
        self.filter_type = filter_type

    def load_pdf(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        return documents
    
    def chunk_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "!", "?", " ", ""],
            is_separator_regex=False,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    

    def create_vectore_store(self, chunks):
        docsearch = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
        )
        if self.filter_type == "pages":
            retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={
                "k": 15,
                "filter": {'page': {'$in': self.filter_value}}
                })
        else:
            retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        return retriever
    
    def __call__(self):
        documents = self.load_pdf()
        chunks = self.chunk_documents(documents)
        retriever = self.create_vectore_store(chunks)
        return retriever
    

# ret = VectoreStore(file_path="Stages_of_data_mining.pdf")
# retriever = ret() 
# query = "What are the main stages of data mining?"
# docs = retriever.invoke(query)
# print(f"\n--- Found {len(docs)} relevant chunks ---\n")
# for i, doc in enumerate(docs):
#     print(f"Chunk {i+1}:")
#     print(f"Content: {doc.page_content}") 
#     # print(f"Source: {doc.metadata}\n")