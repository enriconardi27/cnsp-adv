import streamlit as st
import os
import json
import shutil
import csv
import pandas as pd
import docx
import zipfile
import py7zr
from pptx import Presentation
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "INSERISCI_LA_TUA_API_KEY_QUI")
except (AttributeError, FileNotFoundError):
    GOOGLE_API_KEY = "INSERISCI_LA_TUA_API_KEY_QUI"

VECTOR_STORE_PATH = "vector_store"
USER_CHATS_PATH = "user_chats"
TEMP_FILES_PATH = "temp"
FEEDBACK_FILE_PATH = "feedback_log.csv"

def get_user_data_path(user_id):
    return os.path.join(USER_CHATS_PATH, f"{user_id}_data.json")

def load_user_data(user_id):
    user_data_path = get_user_data_path(user_id)
    if os.path.exists(user_data_path):
        with open(user_data_path, "r") as f:
            return json.load(f)
    return {"chats": {}, "active_chat_id": None}

def save_user_data(user_id):
    if "user_data" in st.session_state:
        with open(get_user_data_path(user_id), "w") as f:
            json.dump(st.session_state.user_data, f)

def switch_chat(chat_id):
    st.session_state.user_data["active_chat_id"] = chat_id
    active_chat = st.session_state.user_data["chats"][chat_id]
    st.session_state.messages = active_chat["messages"]
    st.session_state.chat_history_tuples = [(msg["content"], st.session_state.messages[i+1]["content"]) for i, msg in enumerate(st.session_state.messages) if msg["role"] == "user" and i+1 < len(st.session_state.messages)]

def create_new_chat():
    chat_id = f"chat_{int(datetime.now().timestamp())}"
    chat_name = f"Nuova Conversazione"
    st.session_state.user_data["chats"][chat_id] = {"name": chat_name, "messages": []}
    switch_chat(chat_id)
def get_chat_summary(user_message, assistant_message):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.0)
        
        prompt = f"""
        Basandoti sulla seguente conversazione, crea un titolo molto breve (massimo 5 parole) che ne riassuma l'argomento principale.

        CONVERSAZIONE:
        Utente: "{user_message}"
        Assistente: "{assistant_message}"

        TITOLO BREVE:
        """
        response = llm.invoke(prompt)
        summary = response.content.strip().replace('"', '')
        return summary
    except Exception as e:
        print(f"Error generating chat summary: {e}")
        return f"Chat del {datetime.now().strftime('%d/%m')}"
        
def get_suggestions(user_question, retriever):
    try:
        similar_docs = retriever.get_relevant_documents(user_question)
        if not similar_docs:
            return None

        context = "\n\n".join([doc.page_content for doc in similar_docs])
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.0)
        suggestion_prompt = f"""
        Un utente ha cercato "{user_question}" ma non ha trovato una risposta diretta.
        Basandoti sul seguente contesto estratto dai documenti, suggerisci 3 argomenti o termini di ricerca alternativi e pertinenti.
        Rispondi solo con una lista puntata di suggerimenti in italiano.

        Contesto:
        {context}

        Suggerimenti:
        """
        response = llm.invoke(suggestion_prompt)
        return response.content
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return None
        
def handle_user_input(user_question):
    if st.session_state.conversation:
        active_chat_id = st.session_state.user_data["active_chat_id"]
        if not active_chat_id:
            st.warning("Nessuna chat attiva. Per favore, crea una nuova chat.")
            return

        response = st.session_state.conversation({'question': user_question, 'chat_history': st.session_state.get('chat_history_tuples', [])})
        if not response['source_documents']:
            suggestions = get_suggestions(user_question, st.session_state.conversation.retriever)
            if suggestions:
                answer = f"Non ho trovato risultati diretti per '{user_question}'.\n\nForse cercavi:\n{suggestions}"
            else:
                answer = f"Mi dispiace, non ho trovato alcuna informazione relativa a '{user_question}' nei documenti a mia disposizione."
            
            citations = []
        else:
            answer = response['answer']
            st.session_state.chat_history_tuples.append((user_question, answer))
            
            citations = []
            for doc in response['source_documents']:
                meta = doc.metadata
                source = meta.get('source', 'N/D')
                citation_detail = f"**{source}**"
                if 'page' in meta: citation_detail += f" (Pagina: {meta['page'] + 1})"
                if 'row' in meta: citation_detail += f" (Riga: {meta['row']})"
                if 'paragraph' in meta: citation_detail += f" (Paragrafo: {meta['paragraph']})"
                if 'slide' in meta: citation_detail += f" (Slide: {meta['slide']})"
                if citation_detail not in citations: citations.append(citation_detail)
        
        assistant_message = {"role": "assistant", "content": answer, "question": user_question, "citations": citations}
        st.session_state.user_data["chats"][active_chat_id]["messages"].append(assistant_message)
        st.session_state.messages = st.session_state.user_data["chats"][active_chat_id]["messages"]
        
        active_chat = st.session_state.user_data["chats"][active_chat_id]
        if len(active_chat["messages"]) == 2:
            new_title = get_chat_summary(user_question, answer)
            if new_title:
                st.session_state.user_data["chats"][active_chat_id]["name"] = new_title

def get_documents_with_detailed_metadata(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_FILES_PATH, uploaded_file.name)
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
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
            except Exception as e: st.error(f"Errore estrazione archivio {uploaded_file.name}: {e}")
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
                    if hasattr(shape, "text"): slide_text += shape.text + "\n"
                if slide_text.strip(): documents.append(Document(page_content=slide_text, metadata={'source': source_display, 'slide': i + 1}))
        elif file_extension == '.csv':
            df = pd.read_csv(temp_file_path)
            for i, row in df.iterrows(): documents.append(Document(page_content=", ".join(f"{col}: {val}" for col, val in row.items()), metadata={'source': source_display, 'row': i + 1}))
        elif file_extension == '.xlsx':
            df = pd.read_excel(temp_file_path)
            for i, row in df.iterrows(): documents.append(Document(page_content=", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val)), metadata={'source': source_display, 'row': i + 1}))
    except Exception as e: st.warning(f"Impossibile processare {file_name}: {e}")
    return documents

def check_password():
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    if st.session_state.password_correct: return True
    st.title("Accesso Riservato - Consip Advisor")
    password = st.text_input("Inserisci la password per accedere", type="password")
    if st.button("Login"):
        try:
            if password == st.secrets.get("APP_PASSWORD", "password123"):
                st.session_state.password_correct = True; st.rerun()
            else: st.error("Password non corretta.")
        except Exception: st.error("Errore configurazione: Password non trovata.")
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
        
        template = """
        Sei un assistente chiamato Consip Advisor. Il tuo compito è rispondere alle domande basandoti esclusivamente sul contesto fornito.
        Rispondi sempre e solo in italiano. Se non conosci la risposta dal contesto, di' "Mi dispiace, non ho trovato informazioni a riguardo nei documenti.".
        Non inventare informazioni.

        Contesto: {context}
        Cronologia Chat: {chat_history}
        Domanda: {question}
        Risposta:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=vector_store.as_retriever(), 
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        return chain
    except Exception as e: st.error(f"Errore creazione catena conversazionale: {e}"); return None

def save_feedback(feedback_data):
    file_exists = os.path.exists(FEEDBACK_FILE_PATH)
    with open(FEEDBACK_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'user_id', 'question', 'response', 'rating', 'comment'])
        if not file_exists: writer.writeheader()
        writer.writerow(feedback_data)

def render_main_app():
    st.title("💡 Consip Advisor")
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
        
        st.markdown("---")
        
        st.header("Le tue Conversazioni")
        if st.button("➕ Nuova Chat"):
            create_new_chat()
            save_user_data(st.session_state.current_user)
        
        chats = st.session_state.user_data.get("chats", {})
        sorted_chats = sorted(chats.items(), key=lambda item: item[0], reverse=True)
        for chat_id, chat_data in sorted_chats:
            if st.button(chat_data["name"], key=chat_id, use_container_width=True):
                switch_chat(chat_id)
        
        st.markdown("---")
        st.header("Base di Conoscenza Condivisa")
        
        allowed_types = ['pdf', 'docx', 'csv', 'xlsx', 'zip', '7z', 'pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
        uploaded_files = st.file_uploader("Carica file o archivi", accept_multiple_files=True, type=allowed_types)

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
                    else: st.warning("Nessun documento valido trovato.")
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
        active_chat_id = st.session_state.user_data.get("active_chat_id")
        if active_chat_id:
            user_message = {"role": "user", "content": user_question}
            st.session_state.user_data["chats"][active_chat_id]["messages"].append(user_message)
            st.session_state.messages = st.session_state.user_data["chats"][active_chat_id]["messages"]
            
            with st.chat_message("user"): st.markdown(user_question)
            
            with st.spinner("Consip Advisor sta pensando..."):
                handle_user_input(user_question)
            
            save_user_data(st.session_state.current_user)
            st.rerun()
        else:
            st.warning("Per favore, crea una 'Nuova Chat' per iniziare.")

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
            if 'user_data' not in st.session_state:
                st.session_state.user_data = load_user_data(st.session_state.current_user)
                if not st.session_state.user_data.get("active_chat_id") or not st.session_state.user_data["chats"]:
                    create_new_chat()
                else:
                    switch_chat(st.session_state.user_data["active_chat_id"])
            
            render_main_app()


