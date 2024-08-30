{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9e200f68-ea53-45d5-9a7e-d74c6a975cd9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\z.mokri\\AppData\\Local\\anaconda3\\envs\\CCI\\Lib\\site-packages\\langchain_core\\_api\\deprecation.py:151: LangChainDeprecationWarning: The class `OpenAIEmbeddings` was deprecated in LangChain 0.0.9 and will be removed in 0.3.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import OpenAIEmbeddings`.\n",
      "  warn_deprecated(\n",
      "C:\\Users\\z.mokri\\AppData\\Local\\anaconda3\\envs\\CCI\\Lib\\site-packages\\langchain_core\\_api\\deprecation.py:151: LangChainDeprecationWarning: The class `OpenAI` was deprecated in LangChain 0.0.10 and will be removed in 0.3.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import OpenAI`.\n",
      "  warn_deprecated(\n",
      "2024-08-29 15:17:23.211 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-08-29 15:17:23.347 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\z.mokri\\AppData\\Local\\anaconda3\\envs\\CCI\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-08-29 15:17:23.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-08-29 15:17:23.348 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-08-29 15:17:23.349 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-08-29 15:17:23.351 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-08-29 15:17:23.352 Session state does not function when running a script without `streamlit run`\n",
      "2024-08-29 15:17:23.355 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-08-29 15:17:23.356 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import streamlit as st\n",
    "from langchain.embeddings import OpenAIEmbeddings\n",
    "from langchain.vectorstores import FAISS\n",
    "from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader\n",
    "from langchain.chains import RetrievalQA\n",
    "from langchain.llms import OpenAI\n",
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "from langchain.schema import Document\n",
    "\n",
    "# Configuration de l'API OpenAI\n",
    "openai_api_key = \"sk-tEGcP0ez_Xuf-Grr9KP_OKaC-6M006UnOR28nrE_U9T3BlbkFJIcbflkGDUjX4bV7bsJz_RoCoLzwx_8_iHGMwo88uIA\"\n",
    "embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)\n",
    "\n",
    "# Répertoire contenant les documents\n",
    "directory = \"C:/Users/z.mokri/Downloads/CCI\"\n",
    "\n",
    "# Fonction pour charger les documents\n",
    "def load_documents_from_directory(directory):\n",
    "    documents = []\n",
    "    for filename in os.listdir(directory):\n",
    "        file_path = os.path.join(directory, filename)\n",
    "        if filename.endswith(\".pdf\"):\n",
    "            loader = PyPDFLoader(file_path)\n",
    "            documents.extend(loader.load())\n",
    "        elif filename.endswith(\".docx\"):\n",
    "            loader = UnstructuredWordDocumentLoader(file_path)\n",
    "            documents.extend(loader.load())\n",
    "        elif filename.endswith(\".xlsx\"):\n",
    "            loader = UnstructuredExcelLoader(file_path)\n",
    "            documents.extend(loader.load())\n",
    "    return documents\n",
    "\n",
    "# Charger tous les documents du répertoire\n",
    "documents = load_documents_from_directory(directory)\n",
    "\n",
    "# Convertir les contenus en objets Document\n",
    "doc_objects = []\n",
    "for doc in documents:\n",
    "    if isinstance(doc, str):\n",
    "        doc_objects.append(Document(page_content=doc))\n",
    "    elif hasattr(doc, 'text'):\n",
    "        doc_objects.append(Document(page_content=doc.text))\n",
    "    else:\n",
    "        doc_objects.append(Document(page_content=str(doc)))\n",
    "\n",
    "# Créer un splitter pour diviser le texte en morceaux plus petits\n",
    "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n",
    "\n",
    "# Diviser les documents chargés en morceaux plus petits\n",
    "split_documents = [text_splitter.split_text(doc.page_content) for doc in doc_objects]\n",
    "\n",
    "# Aplatir la liste de listes en une seule liste de morceaux de texte\n",
    "flattened_documents = [item for sublist in split_documents for item in sublist]\n",
    "\n",
    "# Créer une base de données FAISS pour la recherche\n",
    "faiss_documents = [Document(page_content=chunk) for chunk in flattened_documents]\n",
    "db = FAISS.from_documents(faiss_documents, embeddings)\n",
    "\n",
    "# Configurer une chaîne de récupération avec un modèle OpenAI\n",
    "qa_chain = RetrievalQA.from_chain_type(\n",
    "    llm=OpenAI(openai_api_key=\"sk-tEGcP0ez_Xuf-Grr9KP_OKaC-6M006UnOR28nrE_U9T3BlbkFJIcbflkGDUjX4bV7bsJz_RoCoLzwx_8_iHGMwo88uIA\"),\n",
    "    retriever=db.as_retriever()\n",
    ")\n",
    "\n",
    "st.title(\"Chatbot CCI\")\n",
    "question = st.text_input(\"Posez votre question ici:\")\n",
    "\n",
    "if question:\n",
    "    response = qa_chain.run(query=question, max_tokens=100)\n",
    "    st.write(\"Réponse:\", response)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f6688192-b069-4cbd-882c-45055e5f37dd",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (74927840.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[3], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit run Streamlit.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit run Streamlit.py\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "0efbeb1f-8299-4427-a298-adcc6bec357e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
