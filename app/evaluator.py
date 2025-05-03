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
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from typing import Dict
import json
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audiobook_companion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

class GeminiEvaluator:
    def __init__(self, book_title: str):
        self.book_title = book_title
        self.evaluation_history = []
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.error("Gemini API key not found in environment variables")
            raise ValueError("Gemini API key is missing")
        
        genai.configure(api_key=self.gemini_api_key)
        self.model_name = "gemini-1.5-pro"
        try:
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized GeminiEvaluator with model: {self.model_name} for book: {book_title}")
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {e}")
            self._fallback_to_available_model()

    def _fallback_to_available_model(self):
        """Attempt to select an available model if the default fails"""
        try:
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            logger.info(f"Available models: {available_models}")
            if available_models:
                self.model_name = available_models[0].split('/')[-1]  # Extract model name
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Fallback to model: {self.model_name}")
            else:
                logger.error("No models available for generateContent")
                raise ValueError("No supported models available")
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise

    async def evaluate_response(self, user_query: str, assistant_response: str) -> Dict:
        logger.info(f"Evaluating response for query: {user_query}")
        prompt = f"""
        You are an expert in evaluating AI-generated responses in a conversational agent.
        Evaluate the assistant's response to a user's query from a book companion agent.

        User Query: "{user_query}"
        Assistant Response: "{assistant_response}"

        Score the response on:
        1. Relevance to the query (0-5)
        2. Accuracy of information (0-5)
        3. Clarity and helpfulness (0-5)
        4. Depth of response (0-5)
        5. Usefulness to the user (0-5)

        Provide a brief overall feedback comment.

        Return in the following JSON format:
        {{
           "relevance": <int>,
           "accuracy": <int>,
           "clarity": <int>,
           "depth": <int>,
           "usefulness": <int>,
           "overall": <int>,
           "feedback": "<string>"
        }}
        """
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            raw_text = response.text.strip()
            logger.debug(f"Raw Gemini response: {raw_text}")

            # Remove code fences if present
            if raw_text.startswith('```json') and raw_text.endswith('```'):
                raw_text = raw_text[7:-3].strip()

            # Parse JSON response
            try:
                evaluation = json.loads(raw_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {raw_text}")
                evaluation = {
                    "relevance": 0,
                    "accuracy": 0,
                    "clarity": 0,
                    "depth": 0,
                    "usefulness": 0,
                    "overall": 0,
                    "feedback": "Error: Unable to parse evaluation response."
                }

            # Validate evaluation structure
            required_keys = {'relevance', 'accuracy', 'clarity', 'depth', 'usefulness', 'overall', 'feedback'}
            if not all(key in evaluation for key in required_keys):
                missing = required_keys - set(evaluation.keys())
                logger.warning(f"Evaluation missing required keys: {missing}")
                for key in missing:
                    evaluation[key] = 0 if key != 'feedback' else "Error: Missing evaluation field."

            # Ensure scores are integers and within valid range
            for key in ['relevance', 'accuracy', 'clarity', 'depth', 'usefulness', 'overall']:
                try:
                    evaluation[key] = max(0, min(5, int(evaluation[key])))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {key}: {evaluation[key]}")
                    evaluation[key] = 0

            logger.info(f"Evaluation result: {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating response: {e}", exc_info=True)
            return {
                "relevance": 0,
                "accuracy": 0,
                "clarity": 0,
                "depth": 0,
                "usefulness": 0,
                "overall": 0,
                "feedback": f"Error during evaluation: {str(e)}"
            }

    def generate_evaluation_report(self) -> str:
        logger.info("Generating evaluation report")
        if not self.evaluation_history:
            logger.info("No evaluations to report")
            return "No substantive responses were evaluated during this session."

        total_responses = len(self.evaluation_history)
        avg_scores = {
            'relevance': round(sum(e['relevance'] for e in self.evaluation_history) / total_responses, 1),
            'accuracy': round(sum(e['accuracy'] for e in self.evaluation_history) / total_responses, 1),
            'clarity': round(sum(e['clarity'] for e in self.evaluation_history) / total_responses, 1),
            'depth': round(sum(e['depth'] for e in self.evaluation_history) / total_responses, 1),
            'usefulness': round(sum(e['usefulness'] for e in self.evaluation_history) / total_responses, 1),
            'overall': round(sum(e['overall'] for e in self.evaluation_history) / total_responses, 1)
        }

        best_response = max(self.evaluation_history, key=lambda x: x['overall'])
        worst_response = min(self.evaluation_history, key=lambda x: x['overall'])

        report = [
            "\n📊 ASSISTANT RESPONSE EVALUATION REPORT",
            f"📖 Book: {self.book_title}",
            f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🔢 Total Evaluated Responses: {total_responses}",
            "",
            "📈 Average Scores:",
            f"  • Relevance: {avg_scores['relevance']}/5",
            f"  • Accuracy: {avg_scores['accuracy']}/5",
            f"  • Clarity: {avg_scores['clarity']}/5",
            f"  • Depth: {avg_scores['depth']}/5",
            f"  • Usefulness: {avg_scores['usefulness']}/5",
            f"  • Overall: {avg_scores['overall']}/5",
            "",
            f"🏆 Best Response (Score: {best_response['overall']}/5):",
            f"  • User Query: {best_response['question']}",
            f"  • Assistant: {best_response['response'][:150]}...",
            f"  • Feedback: {best_response['feedback']}",
            "",
            f"⚠️ Needs Improvement (Score: {worst_response['overall']}/5):",
            f"  • User Query: {worst_response['question']}",
            f"  • Assistant: {worst_response['response'][:150]}...",
            f"  • Feedback: {worst_response['feedback']}",
            "",
            "🔍 Detailed Breakdown:"
        ]

        for i, eval in enumerate(self.evaluation_history, 1):
            report.extend([
                "",
                f"{i}. User: {eval['question']}",
                f"   Assistant: {eval['response'][:100]}...",
                f"   • Overall: {eval['overall']}/5",
                f"   • Breakdown: R{eval['relevance']} A{eval['accuracy']} C{eval['clarity']} D{eval['depth']} U{eval['usefulness']}",
                f"   • Feedback: {eval['feedback']}"
            ])

        report.extend([
            "",
            "📝 Summary:",
            f"The assistant provided {total_responses} substantive responses about '{self.book_title}'.",
            f"Average quality score was {avg_scores['overall']}/5.",
            f"Areas of strength: {', '.join(k for k, v in avg_scores.items() if v == max(avg_scores.values()))}",
            f"Areas for improvement: {', '.join(k for k, v in avg_scores.items() if v == min(avg_scores.values()))}"
        ])

        report_text = "\n".join(report)
        logger.info("Evaluation report generated")
        return report_text

def process_pdf_to_pinecone(pdf_path: str):
    logger.info(f"Processing PDF: {pdf_path}")
    try:
        book_title = os.path.splitext(os.path.basename(pdf_path))[0]
        namespace = book_title.lower().replace(" ", "-").replace(".", "").replace(",", "")
        logger.info(f"Book title: {book_title}, Namespace: {namespace}")

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

        vectorstore = LangchainPinecone.from_existing_index(index_name, embeddings)

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            success = False
            retries = 3
            while not success and retries > 0:
                try:
                    vectorstore.add_documents(batch_docs, namespace=namespace)
                    success = True
                    logger.info(f"Indexed batch {i}-{i+batch_size}")
                except Exception as e:
                    retries -= 1
                    logger.error(f"Error indexing batch {i}-{i+batch_size}: {e}")
                    if retries > 0:
                        logger.info(f"Retrying in 120 seconds... ({retries} retries left)")
                        time.sleep(120)
            if not success:
                logger.error(f"Failed to index batch {i}-{i+batch_size} after retries")

        console.print(f"✅ Upload complete! {len(documents)} chunks indexed for '{book_title}' in namespace: {namespace}", style="green")
        logger.info(f"Completed indexing {len(documents)} chunks")
        return namespace, book_title
    except Exception as e:
        logger.error(f"PDF Processing Error: {e}")
        console.print(f"[PDF Processing Error] {e}", style="red")
        return None, None

async def get_relevant_context(query: str, namespace: str, top_k: int = 3) -> str:
    logger.info(f"Fetching context for query: {query}")
    try:
        query_embedding = embeddings.embed_query(query)
        if not query_embedding:
            logger.error("Failed to get query embedding")
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
                logger.info(f"Retrieved context: {context[:100]}...")
                return context.strip()
            except Exception as e:
                retries -= 1
                logger.error(f"Query Retry Error: {e}")
                console.print(f"[Query Retry Error] {e}", style="red")
                await asyncio.sleep(3)

        logger.error("Could not fetch context after retries")
        console.print(f"[Query Failure] Could not fetch context after retries", style="red")
        return ""
    except Exception as e:
        logger.error(f"Context Error: {e}")
        console.print(f"[Context Error] {e}", style="red")
        return ""

async def assistant_chat(messages, namespace: str, book_title: str, model='llama3-8b-8192', min_duration=3):
    logger.info("Processing assistant chat")
    start_time = datetime.now()
    user_query = messages[-1]['content'] if messages[-1]['role'] == 'user' else ""

    try:
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
        logger.info(f"Assistant response: {response[:100]}...")
        return response
    except Exception as e:
        logger.error(f"Assistant chat error: {e}")
        raise

def text_to_speech(text):
    logger.info("Converting text to speech")
    try:
        headers = {
            'Authorization': f'Token {deepgram_api_key}',
            'Content-Type': 'application/json'
        }
        formatted_text = text.replace('.', '. ').replace('?', '? ').replace('!', '! ')
        res = requests.post(DEEPGRAM_TTS_URL, headers=headers, json={'text': formatted_text}, stream=True)
        if res.status_code != 200:
            logger.error(f"TTS Error {res.status_code}: {res.text}")
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
        logger.info("TTS playback completed")
    except Exception as e:
        logger.error(f"TTS playback error: {e}")
        console.print(f"TTS playback error: {e}", style="red")

def check_audio_devices():
    logger.info("Checking audio devices")
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
            logger.error("No input audio devices found")
            console.print("[red]No input audio devices found. Please check your microphone.[/red]")
            return None
        console.print(f"[green]Found {len(input_devices)} input devices: {[d['name'] for d in input_devices]}[/green]")
        console.print(f"[green]Using default device: {input_devices[0]['name']} (index: {default_device_index})[/green]")
        logger.info(f"Using default audio device: {input_devices[0]['name']}")
        return default_device_index
    except Exception as e:
        logger.error(f"Error checking audio devices: {e}")
        console.print(f"[red]Error checking audio devices: {e}[/red]")
        return None

def record_voice_input(duration=10, rate=16000):
    logger.info("Recording voice input")
    device_index = check_audio_devices()
    if device_index is None:
        logger.error("No audio device available for recording")
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
            logger.warning("No audio data recorded")
            console.print("[yellow]No audio data recorded. Please try again.[/yellow]")
            return None

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if np.max(np.abs(audio_array)) < 500:
            logger.warning("Audio too quiet, likely no speech detected")
            console.print("[yellow]Audio too quiet, likely no speech detected.[/yellow]")
            return None

        logger.info("Voice input recorded successfully")
        return audio_data
    except Exception as e:
        logger.error(f"Recording error: {e}")
        console.print(f"Recording error: {e}", style="red")
        return None

def transcribe_audio(audio_data, sample_rate=16000):
    logger.info("Transcribing audio")
    try:
        if not audio_data:
            logger.warning("No audio data provided for transcription")
            console.print("[yellow]No audio data provided for transcription.[/yellow]")
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
                        logger.warning("No speech detected in audio")
                        console.print("[yellow]No speech detected in audio.[/yellow]")
                    logger.info(f"Transcription: {transcript}")
                    return transcript
                else:
                    logger.error(f"Transcription error {response.status_code}: {response.text}")
                    console.print(f"Transcription error {response.status_code}: {response.text}", style="red")
                    retries -= 1
                    if retries > 0:
                        logger.info(f"Retrying transcription... ({retries} retries left)")
                        console.print(f"Retrying transcription... ({retries} retries left)", style="yellow")
                        time.sleep(2)
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                console.print(f"Transcription error: {e}", style="red")
                retries -= 1
                if retries > 0:
                    logger.info(f"Retrying transcription... ({retries} retries left)")
                    console.print(f"Retrying transcription... ({retries} retries left)", style="yellow")
                    time.sleep(2)
        logger.error("Transcription failed after retries")
        console.print("[red]Transcription failed after retries.[/red]")
        return ""
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        console.print(f"Transcription error: {e}", style="red")
        return ""

async def run(namespace: str, book_title: str):
    logger.info(f"Starting main loop for book: {book_title}")
    try:
        evaluator = GeminiEvaluator(book_title)
    except ValueError as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        console.print(f"[red]Error: {e}. Please check your Gemini API key.[/red]")
        return

    system_message = {'role': 'system', 'content': SYSTEM_PROMPT}
    messages = [system_message]
    memory_size = 10

    # Test evaluation system
    logger.info("Running test evaluation")
    test_eval = await evaluator.evaluate_response(
        "What is the main theme of this book?",
        "The main theme explores the decline of the American Dream through the tragic story of Jay Gatsby."
    )
    console.print(f"Test evaluation result: {test_eval}", style="cyan")
    logger.info(f"Test evaluation result: {test_eval}")

    try:
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
                logger.info(f"User input: {user_input}")

                messages.append({'role': 'user', 'content': user_input})
                if len(messages) > memory_size:
                    messages = [system_message] + messages[-(memory_size - 1):]

                assistant_response = await assistant_chat(messages, namespace, book_title)
                messages.append({'role': 'assistant', 'content': assistant_response})
                console.print(f"Assistant: {assistant_response}", style="blue")
                logger.info(f"Assistant response: {assistant_response[:100]}...")
                text_to_speech(assistant_response)

                # Evaluate response
                evaluation = await evaluator.evaluate_response(user_input, assistant_response)
                evaluator.evaluation_history.append({
                    "question": user_input,
                    "response": assistant_response,
                    **evaluation
                })
                
                # Print immediate feedback
                console.print(f"\n[Evaluation] Overall: {evaluation['overall']}/5 - {evaluation['feedback']}", style="magenta")
                logger.info(f"Evaluation: Overall {evaluation['overall']}/5 - {evaluation['feedback']}")

            except Exception as e:
                logger.error(f"Runtime error: {e}", exc_info=True)
                console.print(f"Runtime error: {e}", style="red")
                continue

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        console.print("\n\n" + "="*50, style="bold blue")
        console.print("📝 GENERATING FINAL EVALUATION REPORT", style="bold green")
        console.print("="*50, style="bold blue")

        report = evaluator.generate_evaluation_report()
        console.print(report, style="green")

        report_filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        console.print(f"\nReport saved to {report_filename}", style="green")
        logger.info(f"Report saved to {report_filename}")

if __name__ == "__main__":
    logger.info("Starting audiobook companion")
    try:
        pdf_path = input("Enter the path to the PDF book (e.g., C:/path/to/book.pdf): ")
        if not os.path.exists(pdf_path):
            logger.error(f"File does not exist: {pdf_path}")
            console.print(f"[red]Error: The file '{pdf_path}' does not exist.[/red]")
        else:
            namespace, book_title = process_pdf_to_pinecone(pdf_path)
            if namespace and book_title:
                asyncio.run(run(namespace, book_title))
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        console.print("\nProgram terminated by user.", style="red")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        console.print(f"Unexpected error: {e}", style="red")