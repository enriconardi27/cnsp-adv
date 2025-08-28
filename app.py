import streamlit as st
import os
import json
import shutil
import csv
import pandas as pd
import docx
import zipfile
import py7zr
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from pptx import Presentation

try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyDCuewRX1uqYMWrVApGtcRe8v-t5IPxgO0")
except (AttributeError, FileNotFoundError):
    GOOGLE_API_KEY = "AIzaSyDCuewRX1uqYMWrVApGtcRe8v-t5IPxgO0"

VECTOR_STORE_PATH = "vector_store"
USER_CHATS_PATH = "user_chats"
TEMP_FILES_PATH = "temp"
FEEDBACK_FILE_PATH = "feedback_log.csv"

def get_documents_with_detailed_metadata(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_FILES_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_extension = os.path.splitext(uploaded_file.name)[1]

        if file_extension in ['.zip', '.7z']:
            extraction_path = os.path.join(TEMP_FILES_PATH, uploaded_file.name + "_extracted")
            if not os.path.exists(extraction_path): os.makedirs(extraction_path)
            try:
                if file_extension == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zf: zf.extractall(extraction_path)
                elif file_extension == '.7z':
                    with py7zr.SevenZipFile(file_path, mode='r') as z: z.extractall(path=extraction_path)
                
                for root, _, files in os.walk(extraction_path):
                    for file in files:
                        with open(os.path.join(root, file), "rb") as f_in:
                            all_documents.extend(process_single_file(f_in, file, uploaded_file.name))
            except Exception as e:
                st.error(f"Errore durante l'estrazione dell'archivio {uploaded_file.name}: {e}")
        else:
            all_documents.extend(process_single_file(uploaded_file, uploaded_file.name))
    return all_documents

def process_single_file(file_obj, file_name, archive_name=None):
    documents = []
    source_display = f"{archive_name} -> {file_name}" if archive_name else file_name
    
    temp_file_path = os.path.join(TEMP_FILES_PATH, file_name)
    if hasattr(file_obj, 'getbuffer'):
        with open(temp_file_path, "wb") as f: f.write(file_obj.getbuffer())
    
    file_extension = os.path.splitext(file_name)[1]
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()
            for page in pages: page.metadata['source'] = source_display
            documents.extend(pages)

        elif file_extension == '.docx':
            doc = docx.Document(temp_file_path)
            for i, p in enumerate(doc.paragraphs):
                if p.text.strip(): documents.append(Document(page_content=p.text, metadata={'source': source_display, 'paragraph': i + 1}))

        elif file_extension == '.pptx':
            prs = Presentation(temp_file_path)
            for i, slide in enumerate(prs.slides):
                slide_text = ""
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text += shape.text + "\n"
                if slide_text.strip():
                    documents.append(Document(
                        page_content=slide_text,
                        metadata={'source': source_display, 'slide': i + 1}
                    ))

        elif file_extension == '.csv':
            df = pd.read_csv(temp_file_path)
            for i, row in df.iterrows():
                content = ", ".join(f"{col}: {val}" for col, val in row.items())
                documents.append(Document(page_content=content, metadata={'source': source_display, 'row': i + 1}))

        elif file_extension == '.xlsx':
            df = pd.read_excel(temp_file_path)
            for i, row in df.iterrows():
                content = ", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                documents.append(Document(page_content=content, metadata={'source': source_display, 'row': i + 1}))
    except Exception as e:
        st.warning(f"Impossibile processare il file {file_name}: {e}")
    return documents


def check_password():
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if st.session_state.password_correct: return True
    st.title("Accesso Riservato - Consip Advisor")
    password = st.text_input("Inserisci la password per accedere", type="password")
    if st.button("Login"):
        try:
            if password == st.secrets.get("APP_PASSWORD", "CR5DrSFC5wzKE"): 
                st.session_state.password_correct = True
                st.rerun()
            else: st.error("La password inserita non è corretta.")
        except Exception: st.error("Errore di configurazione: Password non trovata.")
    return False

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        return vector_store
    except Exception as e: st.error(f"Errore creazione Vector Store: {e}"); return None

def get_conversational_chain(vector_store):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.3)
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), return_source_documents=True)
    except Exception as e: st.error(f"Errore creazione catena conversazionale: {e}"); return None

def handle_user_input(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question, 'chat_history': st.session_state.get('chat_history_tuples', [])})
        answer = response['answer']
        st.session_state.chat_history_tuples.append((user_question, answer))
        citations = []
        if response['source_documents']:
            for doc in response['source_documents']:
                meta = doc.metadata
                source = meta.get('source', 'N/D')
                citation_detail = f"**{source}**"
                if 'page' in meta: citation_detail += f" (Pagina: {meta['page'] + 1})"
                if 'row' in meta: citation_detail += f" (Riga: {meta['row']})"
                if 'paragraph' in meta: citation_detail += f" (Paragrafo: {meta['paragraph']})"
                if 'slide' in meta: citation_detail += f" (Slide: {meta['slide']})"
                if citation_detail not in citations: citations.append(citation_detail)
        st.session_state.messages.append({"role": "assistant", "content": answer, "question": user_question, "citations": citations})
        save_chat_history(st.session_state.current_user)

def save_feedback(feedback_data):
    file_exists = os.path.exists(FEEDBACK_FILE_PATH)
    with open(FEEDBACK_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'user_id', 'question', 'response', 'rating', 'comment'])
        if not file_exists: writer.writeheader()
        writer.writerow(feedback_data)

def get_user_chat_path(user_id): return os.path.join(USER_CHATS_PATH, f"{user_id}.json")
def save_chat_history(user_id):
    if "messages" in st.session_state:
        with open(get_user_chat_path(user_id), "w") as f: json.dump(st.session_state.messages, f)
def load_chat_history(user_id):
    chat_path = get_user_chat_path(user_id)
    if os.path.exists(chat_path):
        with open(chat_path, "r") as f: return json.load(f)
    return []

def render_main_app():
    st.title("💡 Consip Advisor")
    if "messages" not in st.session_state: st.session_state.messages = load_chat_history(st.session_state.current_user)
    if "chat_history_tuples" not in st.session_state: st.session_state.chat_history_tuples = [(msg["content"], st.session_state.messages[i+1]["content"]) for i, msg in enumerate(st.session_state.messages) if msg["role"] == "user" and i+1 < len(st.session_state.messages)]
    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")) and "conversation" not in st.session_state:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            st.session_state.conversation = get_conversational_chain(vector_store)
        except Exception as e: st.error(f"Impossibile caricare la base di conoscenza: {e}")
    with st.sidebar:
        st.header(f"Ciao, {st.session_state.user_full_name}")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
        if st.button("Nuova Chat"):
            st.session_state.messages, st.session_state.chat_history_tuples = [], []
            chat_path = get_user_chat_path(st.session_state.current_user)
            if os.path.exists(chat_path): os.remove(chat_path)
            st.rerun()
        st.markdown("---")
        st.header("Base di Conoscenza Condivisa")
        allowed_types = [
            'pdf', 'docx', 'csv', 'xlsx', 'zip', '7z', 'pptx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ]
        uploaded_files = st.file_uploader("Carica file o archivi (.zip, .7z)", accept_multiple_files=True, type=['pdf', 'docx', 'csv', 'xlsx', 'zip', '7z'])
        if st.button("Processa e Aggiungi"):
            if uploaded_files:
                with st.spinner("Elaborazione in corso..."):
                    docs = get_documents_with_detailed_metadata(uploaded_files)
                    if docs:
                        chunks = get_text_chunks(docs)
                        vs = get_vector_store(chunks)
                        if vs:
                            st.session_state.conversation = get_conversational_chain(vs)
                            st.success("Base di conoscenza aggiornata!")
                            st.rerun()
                    else:
                        st.warning("Nessun documento valido trovato nei file caricati.")
            else: st.warning("Per favore, carica almeno un file.")
    
    for i, message in enumerate(st.session_state.get('messages', [])):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                citations = message.get("citations", [])
                if citations:
                    if len(citations) > 1:
                        with st.expander("Mostra fonti consultate"):
                            for citation in citations: st.markdown(f"- {citation}")
                    else: st.markdown(f"Fonte: {citations[0]}")
                with st.popover("✏️ Fornisci un Feedback"):
                    with st.form(key=f"feedback_form_{i}"):
                        rating = st.radio("Valutazione:", ("Positivo 👍", "Negativo 👎"), horizontal=True)
                        comment = st.text_area("Commento per migliorare:", height=100)
                        if st.form_submit_button("Invia Feedback"):
                            save_feedback({"timestamp": datetime.now().isoformat(), "user_id": st.session_state.current_user, "question": message.get("question", "N/D"), "response": message["content"], "rating": rating.split(" ")[0], "comment": comment})
                            st.toast("Grazie! Feedback salvato.")
    if user_question := st.chat_input("Fai una domanda sui documenti..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        save_chat_history(st.session_state.current_user)
        with st.chat_message("user"): st.markdown(user_question)
        with st.spinner("Consip Advisor sta pensando..."):
            handle_user_input(user_question)
            st.rerun()

if __name__ == "__main__":
    for path in [TEMP_FILES_PATH, VECTOR_STORE_PATH, USER_CHATS_PATH]:
        if not os.path.exists(path): os.makedirs(path)
    if check_password():
        if not st.session_state.get("current_user"):
             with st.form("user_form"):
                st.title("Identificati")
                nome = st.text_input("Nome")
                cognome = st.text_input("Cognome")
                if st.form_submit_button("Entra"):
                    if nome and cognome:
                        user_id = f"{nome.lower().strip()}_{cognome.lower().strip()}"
                        st.session_state.current_user = user_id
                        st.session_state.user_full_name = f"{nome.strip()} {cognome.strip()}"
                        st.rerun()
                    else: st.error("Nome e cognome sono richiesti.")
        else:
            render_main_app()




