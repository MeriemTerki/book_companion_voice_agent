import asyncio
import requests
import wave
import io
import pyaudio
import os
import time
import numpy as np
from datetime import datetime
from rich.console import Console
from dotenv import load_dotenv
from groq import AsyncGroq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Constants
DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=24000'
DEEPGRAM_STT_URL = 'https://api.deepgram.com/v1/listen'
SYSTEM_PROMPT = """You are an audiobook companion assistant for the uploaded book. Help users explore and understand the content of the book.
Respond in an educational, conversational tone. Always reference content from the book to answer user questions. If relevant, include the book's title or specific details from the text in your responses."""

# Initialize clients
console = Console()
groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere embeddings
embeddings = CohereEmbeddings(
    model="embed-english-v2.0",
    cohere_api_key=cohere_api_key
)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_name = "books-voice-agent"

# Ensure the Pinecone index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def process_pdf_to_pinecone(pdf_path: str):
    try:
        # Extract book title from filename (without extension)
        book_title = os.path.splitext(os.path.basename(pdf_path))[0]
        # Create a namespace from the book title (replace spaces and special chars)
        namespace = book_title.lower().replace(" ", "-").replace(".", "").replace(",", "")

        reader = PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
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
                        "page_content": chunk
                    }
                ))

        vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

        batch_size = 20
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
                    console.print(f"Error indexing batch {i}-{i+batch_size}: {e}", style="red")
                    if retries > 0:
                        console.print(f"Retrying in 120 seconds... ({retries} retries left)", style="yellow")
                        time.sleep(120)
            if not success:
                console.print(f"Failed to index batch {i}-{i+batch_size} after retries.", style="red")

        console.print(f"✅ Upload complete! {len(documents)} chunks indexed for '{book_title}' in namespace: {namespace}", style="green")
        return namespace, book_title
    except Exception as e:
        console.print(f"[PDF Processing Error] {e}", style="red")
        return None, None

async def get_relevant_context(query: str, namespace: str, top_k: int = 3) -> str:
    try:
        query_embedding = embeddings.embed_query(query)
        if not query_embedding:
            console.print("[Embedding Error] Failed to get query embedding.", style="red")
            return ""

        index = pc.Index(name=index_name)

        retries = 3
        while retries > 0:
            try:
                results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)
                context = "\n".join([
                    f"[Page {match['metadata'].get('page_number', 'unknown')}]: {match['metadata'].get('page_content', '')}"
                    for match in results.get("matches", [])
                ])
                return context.strip()
            except Exception as e:
                retries -= 1
                console.print(f"[Query Retry Error] {e}", style="red")
                await asyncio.sleep(3)

        console.print(f"[Query Failure] Could not fetch context after retries", style="red")
        return ""
    except Exception as e:
        console.print(f"[Context Error] {e}", style="red")
        return ""

async def assistant_chat(messages, namespace: str, book_title: str, model='llama3-8b-8192', min_duration=3):
    start_time = datetime.now()
    user_query = messages[-1]['content'] if messages[-1]['role'] == 'user' else ""

    if len(user_query.split()) > 3:
        context = await get_relevant_context(user_query, namespace)
        messages_with_context = [
            {'role': 'system', 'content': SYSTEM_PROMPT + f"\n\nBook: {book_title}\nContext: {context}"},
            *messages[1:]
        ]
        res = await groq.chat.completions.create(
            messages=messages_with_context,
            model=model,
            temperature=0.7,
            max_tokens=1024
        )
        response = res.choices[0].message.content
    else:
        res = await groq.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=1024
        )
        response = res.choices[0].message.content

    elapsed = (datetime.now() - start_time).total_seconds()
    if elapsed < min_duration:
        await asyncio.sleep(min_duration - elapsed)
    return response

def text_to_speech(text):
    try:
        headers = {
            'Authorization': f'Token {deepgram_api_key}',
            'Content-Type': 'application/json'
        }
        formatted_text = text.replace('.', '. ').replace('?', '? ').replace('!', '! ')
        res = requests.post(DEEPGRAM_TTS_URL, headers=headers, json={'text': formatted_text}, stream=True)
        if res.status_code != 200:
            console.print(f"TTS Error {res.status_code}: {res.text}", style="red")
            return

        audio_buffer = io.BytesIO(res.content)
        with wave.open(audio_buffer, 'rb') as wf:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            data = wf.readframes(4096)
            while data:
                stream.write(data)
                data = wf.readframes(4096)
            stream.stop_stream()
            stream.close()
            p.terminate()
    except Exception as e:
        console.print(f"TTS playback error: {e}", style="red")

def check_audio_devices():
    try:
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        input_devices = []
        default_device_index = p.get_default_input_device_info()['index']
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info)
        p.terminate()
        if not input_devices:
            console.print("[red]No input audio devices found. Please check your microphone.[/red]")
            return None
        console.print(f"[green]Found {len(input_devices)} input devices: {[d['name'] for d in input_devices]}[/green]")
        console.print(f"[green]Using default device: {input_devices[0]['name']} (index: {default_device_index})[/green]")
        return default_device_index
    except Exception as e:
        console.print(f"[red]Error checking audio devices: {e}[/red]")
        return None

def record_voice_input(duration=10, rate=16000):
    device_index = check_audio_devices()
    if device_index is None:
        return None
    try:
        console.print("[green]Listening...[/green]")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=1024,
            input_device_index=device_index
        )
        frames = []

        for _ in range(0, int(rate / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = b''.join(frames)
        if len(audio_data) == 0:
            console.print("[yellow]No audio data recorded. Please try again.[/yellow]")
            return None

        # Check audio amplitude
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if np.max(np.abs(audio_array)) < 500:
            console.print("[yellow]Audio too quiet, likely no speech detected.[/yellow]")
            return None

        return audio_data
    except Exception as e:
        console.print(f"Recording error: {e}", style="red")
        return None

def transcribe_audio(audio_data, sample_rate=16000):
    try:
        if not audio_data:
            console.print("[yellow]No audio data provided for transcription.[yellow]")
            return ""

        headers = {
            'Authorization': f'Token {deepgram_api_key}',
            'Content-Type': 'audio/wav',
        }
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        audio_buffer.seek(0)

        retries = 3
        while retries > 0:
            try:
                response = requests.post(
                    DEEPGRAM_STT_URL,
                    headers=headers,
                    data=audio_buffer.read()
                )
                if response.status_code == 200:
                    result = response.json()
                    transcript = result['results']['channels'][0]['alternatives'][0].get('transcript', '')
                    if not transcript:
                        console.print("[yellow]No speech detected in audio.[/yellow]")
                    return transcript
                else:
                    console.print(f"Transcription error {response.status_code}: {response.text}", style="red")
                    retries -= 1
                    if retries > 0:
                        console.print(f"Retrying transcription... ({retries} retries left)", style="yellow")
                        time.sleep(2)
            except Exception as e:
                console.print(f"Transcription error: {e}", style="red")
                retries -= 1
                if retries > 0:
                    console.print(f"Retrying transcription... ({retries} retries left)", style="yellow")
                    time.sleep(2)
        console.print("[red]Transcription failed after retries.[/red]")
        return ""
    except Exception as e:
        console.print(f"Transcription error: {e}", style="red")
        return ""

async def run(namespace: str, book_title: str):
    system_message = {'role': 'system', 'content': SYSTEM_PROMPT}
    messages = [system_message]
    memory_size = 10

    console.print(f"[bold green]Audiobook Companion for '{book_title}' is Ready! Speak now...[/bold green]")

    while True:
        try:
            audio = record_voice_input()
            if not audio:
                continue

            user_input = transcribe_audio(audio)
            if not user_input:
                console.print("[yellow]No speech detected. Please try again.[/yellow]")
                continue

            console.print(f"You (transcribed): {user_input}", style="green")

            messages.append({'role': 'user', 'content': user_input})
            if len(messages) > memory_size:
                messages = [system_message] + messages[-(memory_size - 1):]

            assistant_response = await assistant_chat(messages, namespace, book_title)
            messages.append({'role': 'assistant', 'content': assistant_response})
            console.print(f"Assistant: {assistant_response}", style="blue")
            text_to_speech(assistant_response)

        except KeyboardInterrupt:
            console.print("\nSession ended.", style="red")
            break
        except Exception as e:
            console.print(f"Runtime error: {e}", style="red")
            continue

# Main runner
if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF book (e.g., C:/path/to/book.pdf): ")
    if not os.path.exists(pdf_path):
        console.print(f"[red]Error: The file '{pdf_path}' does not exist.[/red]")
    else:
        namespace, book_title = process_pdf_to_pinecone(pdf_path)
        if namespace and book_title:
            asyncio.run(run(namespace, book_title))