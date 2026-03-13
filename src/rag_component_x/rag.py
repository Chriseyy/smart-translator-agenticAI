import os
import json
from typing import Dict, List, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class DocumentRAG:
    def __init__(self, llm_model: str = "qwen3", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        initalizes the RAG system with embedding model and LLM.
        """
        print(f"RAG: Loading Embedding Model ({embedding_model})...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        
        print(f"RAG: Loading LLM ({llm_model})...")
        self.llm = ChatOllama(model=llm_model)

    def query_document(self, json_path: str, query: str) -> Dict[str, Any]:
        """
        Load json document, create vector store, and answer the query.
        """
        if not os.path.exists(json_path):
            return {"status": "error", "message": f"File not found: {json_path}"}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text_content = []
            if isinstance(data, dict):
                text_content = data.get("rec_texts", [])
                if not text_content:
                    text_content = data.get("original_texts", [])
            elif isinstance(data, list):
                text_content = [str(item) for item in data]

            text_content = [t for t in text_content if t and isinstance(t, str) and t.strip()]
            
            if not text_content:
                return {"status": "error", "message": "No text content found in document (fields 'rec_texts' or 'original_texts' empty?)."}

            docs = [Document(page_content=t) for t in text_content]

            vectorstore = FAISS.from_documents(docs, self.embedding_model)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            template = """Answer the question based only on the following context:
            
            {context}

            Question: {question}

            Answer precisely in the language of the question."""
            
            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(query)
            
            source_chunks = [d.page_content for d in retriever.invoke(query)]
            
            return {
                "status": "success",
                "answer": answer,
                "source_chunks": source_chunks
            }

        except Exception as e:
            print(f"RAG Error: {e}")
            return {"status": "error", "message": str(e)}