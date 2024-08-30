import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configuration de l'API OpenAI
openai_api_key = "sk-tEGcP0ez_Xuf-Grr9KP_OKaC-6M006UnOR28nrE_U9T3BlbkFJIcbflkGDUjX4bV7bsJz_RoCoLzwx_8_iHGMwo88uIA"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Interface utilisateur avec Streamlit
st.title("Chat avec LLM + RAG")

directory = st.text_input("C:/Users/z.mokri/Downloads/CCI")

if st.button("Charger et préparer les documents"):
    # Fonction pour charger les documents
    def load_documents_from_directory(directory):
        documents = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(file_path)
                documents.extend(loader.load())
        return documents

    # Charger tous les documents du répertoire
    documents = load_documents_from_directory(directory)

    # Convertir les contenus en objets Document
    doc_objects = []
    for doc in documents:
        if isinstance(doc, str):
            doc_objects.append(Document(page_content=doc))
        elif hasattr(doc, 'text'):
            doc_objects.append(Document(page_content=doc.text))
        else:
            doc_objects.append(Document(page_content=str(doc)))

    # Créer un splitter pour diviser le texte en morceaux plus petits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Diviser les documents chargés en morceaux plus petits
    split_documents = [text_splitter.split_text(doc.page_content) for doc in doc_objects]

    # Aplatir la liste de listes en une seule liste de morceaux de texte
    flattened_documents = [item for sublist in split_documents for item in sublist]

    # Créer une base de données FAISS pour la recherche
    faiss_documents = [Document(page_content=chunk) for chunk in flattened_documents]
    db = FAISS.from_documents(faiss_documents, embeddings)

    st.success("Documents chargés et préparés avec succès!")

    # Configurer une chaîne de récupération avec un modèle OpenAI
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key="sk-tEGcP0ez_Xuf-Grr9KP_OKaC-6M006UnOR28nrE_U9T3BlbkFJIcbflkGDUjX4bV7bsJz_RoCoLzwx_8_iHGMwo88uIA"),
        retriever=db.as_retriever()
    )

    # Poser une question au modèle
    question = st.text_input("Entrez votre question:")

    if question:
        response = qa_chain.run(query=question, max_tokens=100)
        st.write("Réponse:", response)
