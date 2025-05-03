import streamlit as st
import os
import asyncio
import time
import io
import tempfile
import base64
import requests
import pyaudio
import wave
from datetime import datetime
from dotenv import load_dotenv
from groq import AsyncGroq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file if it exists
load_dotenv()

# Constants
DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=24000'
DEEPGRAM_STT_URL = 'https://api.deepgram.com/v1/listen'
SYSTEM_PROMPT = """You are an audiobook companion assistant for the uploaded book. Help users explore and understand the content of the book.
Respond in an educational, conversational tone. Always reference content from the book to answer user questions. If relevant, include the book's title or specific details from the text in your responses."""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "book_initialized" not in st.session_state:
    st.session_state.book_initialized = False
if "namespace" not in st.session_state:
    st.session_state.namespace = None
if "book_title" not in st.session_state:
    st.session_state.book_title = None
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "api_keys_set" not in st.session_state:
    st.session_state.api_keys_set = False
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False
if "pinecone_initialized" not in st.session_state:
    st.session_state.pinecone_initialized = False

# Helper function to check if API keys are set
def check_api_keys():
    keys = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY")
    }
    
    missing_keys = [k for k, v in keys.items() if not v]
    if missing_keys:
        return False, missing_keys
    return True, []

# Function to initialize clients with API keys
def initialize_clients():
    try:
        st.session_state.groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        st.session_state.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        st.session_state.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        st.session_state.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        # Initialize Cohere embeddings
        st.session_state.embeddings = CohereEmbeddings(
            model="embed-english-v2.0",
            cohere_api_key=st.session_state.cohere_api_key
        )
        
        # Initialize Pinecone client
        st.session_state.pc = Pinecone(api_key=st.session_state.pinecone_api_key)
        st.session_state.index_name = "books-voice-agent"
        
        initialize_pinecone_index()
        
        st.session_state.api_keys_set = True
        return True
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return False

def initialize_pinecone_index():
    try:
        # Check if index already exists
        existing_indexes = st.session_state.pc.list_indexes().names()
        
        if st.session_state.index_name not in existing_indexes:
            with st.spinner("Creating Pinecone index (this may take a minute)..."):
                st.session_state.pc.create_index(
                    name=st.session_state.index_name,
                    dimension=4096,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for index to be ready
                time.sleep(60)
        
        # Test connection to index
        try:
            index = st.session_state.pc.Index(st.session_state.index_name)
            st.session_state.pinecone_initialized = True
        except Exception as e:
            st.error(f"Error connecting to Pinecone index: {str(e)}")
            st.session_state.pinecone_initialized = False
    except Exception as e:
        st.error(f"Error initializing Pinecone index: {str(e)}")
        st.session_state.pinecone_initialized = False

# Custom CSS to style the app
def load_css():
    st.markdown("""
    <style>
        .chat-message {
            padding: 1.5rem; 
            border-radius: 0.5rem; 
            margin-bottom: 1rem; 
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #f0f2f6;
        }
        .chat-message.assistant {
            background-color: #e3f2fd;
        }
        .chat-message .message-content {
            margin-top: 0.5rem;
        }
        .audio-controls {
            display: flex;
            align-items: center;
            margin-top: 1rem;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stButton.record > button {
            background-color: #f44336;
        }
        .file-upload-container {
            border: 2px dashed #aaa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .header-container {
            display: flex;
            align-items: center;
            background-color: #2e7d32;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        .header-icon {
            font-size: 2rem;
            margin-right: 1rem;
        }
        .header-text {
            font-size: 1.5rem;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

def process_pdf_to_pinecone(pdf_file):
    try:
        with st.spinner("Processing book and uploading to Pinecone..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                pdf_path = tmp_file.name

            # Extract book title from filename (without extension)
            book_title = pdf_file.name.split('.')[0]
            
            # Create a namespace from the book title
            namespace = book_title.lower().replace(" ", "-").replace(".", "").replace(",", "")

            reader = PdfReader(pdf_path)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            
            if not pages:
                st.error("Could not extract text from the PDF. Please try another file.")
                os.unlink(pdf_path)
                return None, None
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
            
            documents = []
            for page_num, page_text in enumerate(pages, 1):
                chunks = splitter.split_text(page_text)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "book_title": book_title,
                            "page_number": page_num,
                            "page_content": chunk,
                            "text": chunk  # Adding text key for Pinecone
                        }
                    ))

            # Get the Pinecone index client
            index = st.session_state.pc.Index(st.session_state.index_name)
            
            # Initialize the vectorstore with the index client
            from langchain_pinecone import PineconeVectorStore
            
            try:
                vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=st.session_state.embeddings,
                    text_key="text"
                )
            except Exception as e:
                st.error(f"Error initializing vector store: {str(e)}")
                os.unlink(pdf_path)
                return None, None

            progress_bar = st.progress(0)
            total_batches = (len(documents) + 9) // 10  # Ceiling division
            
            # Using smaller batch size to avoid potential issues
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                success = False
                retries = 3
                while not success and retries > 0:
                    try:
                        vectorstore.add_documents(batch_docs, namespace=namespace)
                        success = True
                    except Exception as e:
                        retries -= 1
                        st.warning(f"Error indexing batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                        if retries > 0:
                            st.info(f"Retrying in 10 seconds... ({retries} retries left)")
                            time.sleep(10)
                if not success:
                    st.error(f"Failed to index batch {i//batch_size + 1}/{total_batches} after retries.")
                
                # Update progress bar
                progress_bar.progress((i + len(batch_docs)) / len(documents))

            # Clean up temporary file
            os.unlink(pdf_path)
            
            st.success(f"Upload complete! {len(documents)} text chunks indexed for '{book_title}'")
            return namespace, book_title
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None

async def get_relevant_context(query, namespace, top_k=3):
    try:
        # Get query embedding
        query_embedding = st.session_state.embeddings.embed_query(query)
        
        # Get Pinecone index
        index = st.session_state.pc.Index(st.session_state.index_name)
        
        # Query Pinecone index
        retries = 3
        while retries > 0:
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
                )
                
                # Extract context from results
                context = "\n".join([
                    f"[Page {match['metadata'].get('page_number', 'unknown')}]: {match['metadata'].get('page_content', '')}"
                    for match in results.get("matches", [])
                ])
                return context.strip()
            except Exception as e:
                retries -= 1
                await asyncio.sleep(3)
        
        return ""
    except Exception as e:
        st.error(f"Error getting context: {str(e)}")
        return ""

async def assistant_chat(messages, namespace, book_title, model='llama3-8b-8192'):
    try:
        # Create a clean copy of messages without any custom fields
        clean_messages = []
        for msg in messages:
            clean_msg = {
                'role': msg['role'],
                'content': msg['content']
            }
            clean_messages.append(clean_msg)
        
        user_query = clean_messages[-1]['content'] if clean_messages[-1]['role'] == 'user' else ""

        if len(user_query.split()) > 3:
            context = await get_relevant_context(user_query, namespace)
            system_prompt = SYSTEM_PROMPT + f"\n\nBook: {book_title}\nContext: {context}"
            
            messages_with_context = [
                {'role': 'system', 'content': system_prompt}
            ]
            
            # Add all non-system messages
            for msg in clean_messages:
                if msg['role'] != 'system':
                    messages_with_context.append(msg)
            
            res = await st.session_state.groq.chat.completions.create(
                messages=messages_with_context,
                model=model,
                temperature=0.7,
                max_tokens=1024
            )
            response = res.choices[0].message.content
        else:
            res = await st.session_state.groq.chat.completions.create(
                messages=clean_messages,
                model=model,
                temperature=0.7,
                max_tokens=1024
            )
            response = res.choices[0].message.content

        return response
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

def text_to_speech(text):
    try:
        headers = {
            'Authorization': f'Token {st.session_state.deepgram_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Format text for better speech
        formatted_text = text.replace('.', '. ').replace('?', '? ').replace('!', '! ')
        
        # Request TTS from Deepgram with timeout
        try:
            res = requests.post(
                DEEPGRAM_TTS_URL, 
                headers=headers, 
                json={'text': formatted_text},
                timeout=(5, 30)  # 5 seconds for connection timeout, 30 seconds for read timeout
            )
            
            if res.status_code != 200:
                st.error(f"TTS Error {res.status_code}: {res.text}")
                return None
            
            # Return audio bytes
            return res.content
        except requests.exceptions.Timeout:
            st.error("Connection to Deepgram TTS API timed out. Please check your internet connection and try again.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to Deepgram TTS API. Please check your internet connection.")
            return None
    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        return None
    

# Add this function to your code
def retry_api_call(func, *args, max_retries=3, initial_delay=1, **kwargs):
    """
    Retry an API call with exponential backoff
    
    Args:
        func: The function to call
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    retries = 0
    delay = initial_delay
    
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            retries += 1
            if retries >= max_retries:
                # If we've exhausted our retries, re-raise the exception
                raise e
            
            # Wait with exponential backoff before retrying
            st.warning(f"API call failed, retrying in {delay} seconds... ({retries}/{max_retries})")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    # This should never be reached due to the exception re-raise above
    return None

# Example usage for Deepgram API calls:
# response = retry_api_call(
#     requests.post,
#     DEEPGRAM_STT_URL,
#     headers=headers,
#     data=audio_buffer.read(),
#     timeout=(5, 30)
# )

# Add this function to handle API fallbacks
def handle_api_unavailable():
    """Handle scenario when API is unavailable"""
    # Temporarily disable voice mode
    was_voice_mode = st.session_state.voice_mode
    st.session_state.voice_mode = False
    
    # Inform the user
    st.error("Voice services are currently unavailable. Temporarily switching to text-only mode.")
    
    # Store the original mode to try again later
    st.session_state.previous_voice_mode = was_voice_mode
    
    # Set a flag to retry after some time
    st.session_state.retry_voice_services = True
    st.session_state.last_retry_time = time.time()

# Then in your main loop, add code to periodically retry voice services
def try_restore_voice_services():
    """Try to restore voice services if they were previously unavailable"""
    if getattr(st.session_state, 'retry_voice_services', False):
        current_time = time.time()
        retry_interval = 5 * 60  # 5 minutes
        
        if current_time - st.session_state.last_retry_time > retry_interval:
            # Test if the API is now available
            try:
                # Simple test call to the API
                response = requests.get(
                    "https://api.deepgram.com/v1/system-status",
                    headers={'Authorization': f'Token {st.session_state.deepgram_api_key}'},
                    timeout=5
                )
                
                if response.status_code == 200:
                    # API is available again, restore voice mode
                    st.session_state.voice_mode = getattr(st.session_state, 'previous_voice_mode', False)
                    st.session_state.retry_voice_services = False
                    st.success("Voice services have been restored!")
                    time.sleep(2)  # Give user time to see the message
                    st.rerun()
                else:
                    # API still unavailable, update retry time
                    st.session_state.last_retry_time = current_time
            except:
                # Still having connection issues
                st.session_state.last_retry_time = current_time

# Call this function in your main app
# Add this right before the end of your main_app function:
# try_restore_voice_services()

def audio_to_base64(audio_bytes):
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        return b64
    return None

def record_audio(duration=10, sample_rate=16000):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        audio_data = b''.join(frames)
        return audio_data, sample_rate
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        return None, None

def transcribe_audio(audio_data, sample_rate=16000):
    try:
        if not audio_data:
            st.warning("No audio data provided for transcription.")
            return ""
        
        headers = {
            'Authorization': f'Token {st.session_state.deepgram_api_key}',
            'Content-Type': 'audio/wav',
        }
        
        # Create WAV file in memory
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        audio_buffer.seek(0)
        
        # Send to Deepgram for transcription with proper timeout settings
        try:
            response = requests.post(
                DEEPGRAM_STT_URL,
                headers=headers,
                data=audio_buffer.read(),
                timeout=(5, 30)  # 5 seconds for connection timeout, 30 seconds for read timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                transcript = result['results']['channels'][0]['alternatives'][0].get('transcript', '')
                if not transcript:
                    st.warning("No speech detected in audio.")
                return transcript
            else:
                st.error(f"Transcription error {response.status_code}: {response.text}")
                return ""
        except requests.exceptions.Timeout:
            st.error("Connection to Deepgram API timed out. Please check your internet connection and try again.")
            return ""
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to Deepgram API. Please check your internet connection.")
            return ""
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""
        


def start_recording():
    if st.session_state.recording:
        st.session_state.recording = False
        return
    
    st.session_state.recording = True
    duration = 10  # seconds
    
    with st.spinner(f"Recording for {duration} seconds..."):
        audio_data, sample_rate = record_audio(duration)
        if audio_data:
            st.session_state.audio_bytes = audio_data
            st.session_state.sample_rate = sample_rate
        else:
            st.warning("No audio recorded.")
    
    st.session_state.recording = False
    if hasattr(st.session_state, 'audio_bytes') and st.session_state.audio_bytes:
        # Use st.rerun() instead of experimental_rerun()
        st.rerun()

def handle_voice_input():
    if st.session_state.audio_bytes:
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(st.session_state.audio_bytes, st.session_state.sample_rate)
        
        if transcript:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": transcript})
            # Clear audio bytes
            st.session_state.audio_bytes = None
            # Return True to indicate new message
            return True
    return False

def get_book_namespaces():
    """Get available book namespaces safely without using describe_stats"""
    try:
        # For simplicity, we'll return the current book namespace if it exists
        namespaces = []
        if st.session_state.namespace:
            namespaces.append(st.session_state.namespace)
            
        return namespaces
    except Exception as e:
        st.error(f"Error listing books: {str(e)}")
        return []

def convert_audio_for_playback(audio_bytes, sample_rate=16000):
    """Convert raw audio bytes to WAV format for playback in Streamlit"""
    if not audio_bytes:
        return None
    
    try:
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        
        wav_buffer.seek(0)
        return wav_buffer.getvalue()
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return None

async def main_app():
    load_css()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-icon">📚</div>
        <div class="header-text">Audiobook Companion Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Setup
    with st.sidebar:
        st.header("Configuration")
        
        # Check if API keys are already set
        keys_ok, missing_keys = check_api_keys()
        
        if not keys_ok:
            st.warning("Please set the following API keys:")
            
            with st.form("api_key_form"):
                if "GROQ_API_KEY" in missing_keys:
                    groq_key = st.text_input("Groq API Key", type="password")
                if "DEEPGRAM_API_KEY" in missing_keys:
                    deepgram_key = st.text_input("Deepgram API Key", type="password")
                if "PINECONE_API_KEY" in missing_keys:
                    pinecone_key = st.text_input("Pinecone API Key", type="password")
                if "COHERE_API_KEY" in missing_keys:
                    cohere_key = st.text_input("Cohere API Key", type="password")
                
                submitted = st.form_submit_button("Save API Keys")
                
                if submitted:
                    # Set environment variables
                    if "GROQ_API_KEY" in missing_keys and groq_key:
                        os.environ["GROQ_API_KEY"] = groq_key
                    if "DEEPGRAM_API_KEY" in missing_keys and deepgram_key:
                        os.environ["DEEPGRAM_API_KEY"] = deepgram_key
                    if "PINECONE_API_KEY" in missing_keys and pinecone_key:
                        os.environ["PINECONE_API_KEY"] = pinecone_key
                    if "COHERE_API_KEY" in missing_keys and cohere_key:
                        os.environ["COHERE_API_KEY"] = cohere_key
                    
                    # Initialize clients
                    success = initialize_clients()
                    if success:
                        st.success("API keys saved successfully!")
                        st.rerun()
        else:
            if not st.session_state.api_keys_set:
                success = initialize_clients()
                if success:
                    st.success("All API keys are set!")
            else:
                st.success("All API keys are set!")
        
        # Model selection
        model_options = {
            "Llama3 8B": "llama3-8b-8192",
            "Llama3 70B": "llama3-70b-8192",
            "Mixtral 8x7B": "mixtral-8x7b-32768"
        }
        selected_model = st.selectbox(
            "Select LLM Model", 
            options=list(model_options.keys()),
            index=0
        )
        st.session_state.model = model_options[selected_model]
        
        # Voice mode toggle
        st.session_state.voice_mode = st.toggle("Voice Mode", st.session_state.voice_mode)
        
        # Book operations
        st.header("Book Management")
        
        if st.session_state.book_initialized:
            st.success(f"Current book: {st.session_state.book_title}")
            if st.button("Clear Current Book"):
                st.session_state.book_initialized = False
                st.session_state.namespace = None
                st.session_state.book_title = None
                st.session_state.messages = []
                st.rerun()
        
        # Book list
        if st.session_state.api_keys_set:
            book_namespaces = get_book_namespaces()
            if book_namespaces:
                st.header("Your Books")
                for ns in book_namespaces:
                    book_name = ns.replace("-", " ").title()
                    if st.button(f"📖 {book_name}", key=f"book_{ns}"):
                        st.session_state.namespace = ns
                        st.session_state.book_title = book_name
                        st.session_state.book_initialized = True
                        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                        st.rerun()
    
    # Main content area
    if not st.session_state.api_keys_set:
        st.info("Please set up your API keys in the sidebar to get started.")
        return
    
    # Book upload area (only shown if no book is initialized)
    if not st.session_state.book_initialized:
        st.subheader("Upload a New Book")
        st.markdown("""
        <div class="file-upload-container">
            <h3>Upload a PDF book to get started</h3>
            <p>The book will be processed and indexed for AI-powered conversations</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            if not st.session_state.pinecone_initialized:
                st.error("Pinecone index not initialized properly. Please check your API keys and try again.")
                return
                
            namespace, book_title = process_pdf_to_pinecone(uploaded_file)
            if namespace and book_title:
                st.session_state.namespace = namespace
                st.session_state.book_title = book_title
                st.session_state.book_initialized = True
                # Initialize with system message
                st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                st.rerun()
    else:
        # Chat interface
        st.subheader(f"Chat with your book: {st.session_state.book_title}")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            
            if message["role"] == "user":
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div><strong>You</strong></div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div><strong>Assistant</strong></div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add audio playback for assistant messages
                    if "audio_b64" not in message:
                        audio_bytes = text_to_speech(message["content"])
                        if audio_bytes:
                            message["audio_b64"] = audio_to_base64(audio_bytes)
                    
                    if "audio_b64" in message and message["audio_b64"]:
                        audio_data = base64.b64decode(message["audio_b64"])
                        st.audio(audio_data, format="audio/wav")
        
        # Voice input handling
        if st.session_state.voice_mode:
            voice_col1, voice_col2 = st.columns([1, 4])
            with voice_col1:
                if st.button("🎤 Record" if not st.session_state.recording else "⏹️ Stop", 
                            key="record_button", 
                            on_click=start_recording):
                    pass
            
            with voice_col2:
                if st.session_state.audio_bytes:
                    # Convert audio bytes to WAV format for playback
                    wav_data = convert_audio_for_playback(
                        st.session_state.audio_bytes,
                        st.session_state.sample_rate
                    )
                    if wav_data:
                        st.audio(wav_data, format="audio/wav")
                    
                    if handle_voice_input():
                        st.rerun()
        
        # Text input
        user_input = st.chat_input("Type your question about the book...")
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()
        
        # Process new messages
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Thinking..."):
                # Make a copy of messages for processing
                messages_copy = st.session_state.messages.copy()
                
                # Get assistant response
                response = await assistant_chat(
                    messages_copy, 
                    st.session_state.namespace, 
                    st.session_state.book_title,
                    model=st.session_state.model
                )
                
                # Generate audio for response
                audio_bytes = text_to_speech(response)
                audio_b64 = audio_to_base64(audio_bytes) if audio_bytes else None
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "audio_b64": audio_b64
                })
                
                st.rerun()

# Create an app directory if it doesn't exist
if not os.path.exists('app'):
    os.makedirs('app')

# Run the app
if __name__ == "__main__":
    asyncio.run(main_app())