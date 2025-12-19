import os
import json
import asyncio
import shutil
import tempfile
import uuid
import base64
import io
from typing import Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import httpx
import motor.motor_asyncio
from fastapi import FastAPI, File, UploadFile, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import faiss
import numpy as np
from datetime import datetime, timezone
from huggingface_hub import InferenceClient
from pypdf import PdfReader


MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ChatHistoryManager:
    def __init__(self, mongo_client):
        self.client = mongo_client
        self.db = self.client["IITI_Tutor_DB"]
        self.collection = self.db["chat_history"]

    async def save_message(self, user_id, conversation_id, role, content):
        msg = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        }
        try:
            await self.collection.insert_one(msg)
        except Exception as e:
            print(f"DB Error: {e}")

    async def load_history(self, user_id, conversation_id, limit=6):
        try:
            cursor = self.collection.find(
                {"user_id": user_id, "conversation_id": conversation_id},
                {"role": 1, "content": 1, "_id": 0}
            ).sort("timestamp", -1).limit(limit)
            history = [doc async for doc in cursor]
            return history[::-1]
        except:
            return []


class BaseLLMBot:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    async def call_groq(self, messages, model="llama-3.1-8b-instant", temp=0.7):
        payload = {"model": model, "messages": messages, "temperature": temp}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(GROQ_API_URL, headers=self.headers, json=payload, timeout=60)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "I'm having trouble connecting to my brain right now. Please try again."



class QueryBot(BaseLLMBot):
    """
    Handles:
    1. General academic queries (RAG).
    2. 'Tutor' mode (Interactive teaching).
    """
    def __init__(self):
        super().__init__()
        self.hf_client = InferenceClient(token=HF_TOKEN)
        self.chunks = []
        self.metadata = []
        self.index = None
        self.is_ready = False

    async def initialize_data(self):
        """Loads curriculum data from MongoDB in background"""
        try:
            print(" [Background] Loading Course Data...")
            if not MONGO_URI:
                print(" MONGO_URI not set. Skipping DB load.")
                self.is_ready = True
                return

            db = mongo_client["IITI_Tutor_DB"]
            collection = db["First_Year_Curriculum"]
            cursor = collection.find({})
            raw_courses = await cursor.to_list(length=None)
            
            self.chunks = []
            self.metadata = []
            
            # Create search chunks from DB data
            for course in raw_courses:
                # Combine relevant fields into a text block
                text_block = f"Course: {course.get('Course Code','')} - {course.get('Course Title','')}\n"
                for k, v in course.items():
                    if k not in ["_id", "Course Code", "Course Title"]:
                        text_block += f"{k}: {v}\n"
                
                # Split large blocks if necessary (simple char split for now)
                if len(text_block) > 50:
                    self.chunks.append(text_block[:1500]) # Limit chunk size
                    self.metadata.append({"title": course.get("Course Title", "Unknown")})

            if self.chunks:
                print(f"⏳ Generating Embeddings for {len(self.chunks)} chunks...")
                loop = asyncio.get_event_loop()
                
                # Generate embeddings in batches
                embeddings = []
                batch_size = 20
                for i in range(0, len(self.chunks), batch_size):
                    batch = self.chunks[i:i+batch_size]
                    try:
                        res = await loop.run_in_executor(None, lambda b=batch: self.hf_client.feature_extraction(b, model=EMBEDDING_MODEL))
                        arr = np.array(res)
                        if len(arr.shape) == 3: arr = np.mean(arr, axis=1)
                        embeddings.append(arr)
                    except Exception as e:
                        print(f"Embedding Batch Error: {e}")

                if embeddings:
                    final_emb = np.concatenate(embeddings, axis=0)
                    self.index = faiss.IndexFlatL2(final_emb.shape[1])
                    self.index.add(final_emb)
                    self.is_ready = True
                    print(" RAG System Ready.")
            else:
                print(" No course data found in DB.")
                self.is_ready = True
        except Exception as e:
            print(f" RAG Init Failed: {e}")
            self.is_ready = True

    async def retrieve(self, query, top_k=4):
        if not self.is_ready or not self.index: return []
        try:
            loop = asyncio.get_event_loop()
            q_emb = await loop.run_in_executor(None, lambda: self.hf_client.feature_extraction(query, model=EMBEDDING_MODEL))
            q_vec = np.array(q_emb)
            if len(q_vec.shape) > 1: q_vec = np.mean(q_vec, axis=0)
            q_vec = q_vec.reshape(1, -1)
            
            D, I = self.index.search(q_vec, top_k)
            return [self.chunks[i] for i in I[0] if i < len(self.chunks)]
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return []

    async def answer(self, query, history, mode="query"):
        if not self.is_ready: return " System is still warming up. Please try again in 30 seconds."
        
        # Retrieve context from curriculum
        relevant_text = await self.retrieve(query)
        context = "\n\n".join(relevant_text)
        
        if mode == "tutor":
            sys_prompt = (
                "You are 'Professor Nexus', an expert, patient AI Tutor. "
                "Do NOT just give the answer. TEACH the concept. "
                "1. Use analogies and simple real-world examples. "
                "2. Break complex topics into steps. "
                "3. Ask a checking question at the end to ensure the student understood."
            )
        else:
            sys_prompt = "You are a helpful academic assistant for IIT Indore. Answer based on the context provided."

        msgs = [{"role": "system", "content": sys_prompt}]
        if history: msgs.extend(history)
        msgs.append({"role": "user", "content": f"Context Info:\n{context}\n\nUser Query: {query}"})
        
        return await self.call_groq(msgs)

class BookBot(BaseLLMBot):
    """
    Handles:
    1. Reading uploaded PDF books.
    2. Answering questions specific to that book.
    """
    async def process_book(self, file_path, query):
        text = ""
        try:
            reader = PdfReader(file_path)
            # Limit pages for performance (since using free tier) sed lyf
            max_pages = 50 
            for i, page in enumerate(reader.pages[:max_pages]):
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        except Exception as e:
            return f" Error reading PDF: {e}"

        if len(text.strip()) < 50: 
            return " The PDF seems to be empty or contains scanned images without text. Please try uploading a text-based PDF or use the 'Solve' feature for images."

        prompt = f"""You are analyzing a book uploaded by the user.
        
        BOOK CONTENT (First {max_pages} pages):
        {text[:25000]} 
        ... [Content Truncated] ...
        
        User Query: {query}
        
        Answer specifically based on the book content above. If the answer isn't there, say so."""

        msgs = [{"role": "user", "content": prompt}]
        return await self.call_groq(msgs, model="llama-3.1-8b-instant")

class QuestionPaperBot(BaseLLMBot):
    """
    Handles:
    1. Solving uploaded papers (Image or PDF).
    2. Generating NEW papers from uploaded books/notes.
    """
    async def extract_text_smart(self, file_path):
        """Smart extraction: Tries text-layer first, falls back to OCR."""
        text = ""
        ext = os.path.splitext(file_path)[1].lower()

        # Fast Text Extraction (PDFs)
        if ext == ".pdf":
            try:
                reader = PdfReader(file_path)
                for page in reader.pages[:20]: # Check first 20 pages
                    t = page.extract_text()
                    if t: text += t + "\n"
            except: pass
        
        # OCR (Images or Scanned PDFs)
        if len(text.strip()) < 100: # If text layer failed or is an image
            try:
                if ext == ".pdf":
                    with ThreadPoolExecutor() as pool:
                        images = convert_from_path(file_path, 300)
                        # OCR only first 5 pages to save time/resources
                        ocr_texts = [pytesseract.image_to_string(img) for img in images[:5]]
                        text = "\n".join(ocr_texts)
                else: # Image file
                    text = pytesseract.image_to_string(Image.open(file_path))
            except Exception as e:
                print(f"OCR Error: {e}")
        
        return text

    async def process_file(self, file_path, mode="answer"):
        
        raw_text = await self.extract_text_smart(file_path)

        if not raw_text or len(raw_text.strip()) < 10: 
            return {"text": " I could not read any text from the file. It might be blurry or password protected.", "pdf_file": None}

        if mode == "generate":
            prompt = f"""Using the content below, generate a NEW Question Paper.
            - Create 3 Sections (A, B, C).
            - Include marks for each question.
            - Content Source:
            {raw_text[:15000]}
            """
            model = "llama-3.3-70b-versatile"
        else:
            prompt = f"""Provide detailed, step-by-step SOLUTIONS for the questions found in this text:
            {raw_text[:15000]}
            """
            model = "llama-3.3-70b-versatile"

        generated_text = await self.call_groq([{"role": "user", "content": prompt}], model=model)
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 50
        
        c.setFont("Helvetica", 11)
        for line in generated_text.split('\n'):
            if y < 50: 
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
            
            # Basic text wrapping
            words = line.split()
            current_line = ""
            for word in words:
                if c.stringWidth(current_line + word) < (width - 100):
                    current_line += word + " "
                else:
                    c.drawString(50, y, current_line)
                    y -= 15
                    current_line = word + " "
            c.drawString(50, y, current_line)
            y -= 15
            
        c.save()
        buffer.seek(0)
        
        return {"text": generated_text, "pdf_file": buffer}

class SchedulerBot(BaseLLMBot):
    """Creates Study Schedules"""
    async def create_schedule(self, prompt):
        msg = [{"role": "user", "content": f"Create a structured, realistic daily study schedule based on this request: {prompt}"}]
        return await self.call_groq(msg)


class RouterAgent(BaseLLMBot):
    def __init__(self, history_manager):
        super().__init__()
        self.history = history_manager
        self.rag = QueryBot()
        self.book = BookBot()
        self.qp = QuestionPaperBot()
        self.sched = SchedulerBot()

    async def classify_intent(self, prompt, has_file):
        """Decides which bot to use based on prompt content and file presence."""
        system_msg = f"""Classify the user intent into exactly one category.
        
        Context:
        - File Uploaded: {has_file}
        
        Categories:
        1. 'tutor': User wants to learn concepts ("teach me", "explain", "how does x work").
        2. 'book': User asks about an uploaded book ("summarize this", "what does chapter 1 say").
        3. 'solve': User wants answers to an uploaded paper ("solve this", "solution", "answer key").
        4. 'generate': User wants a NEW paper created ("create a mock test", "generate questions from this book").
        5. 'schedule': User wants a plan ("timetable", "study plan", "schedule").
        6. 'query': General curriculum questions if no file is present.
        
        User Input: "{prompt}"
        
        Output (ONE WORD ONLY):"""
        
        try:
            category = await self.call_groq([{"role": "user", "content": system_msg}], model="llama-3.1-8b-instant", temp=0.1)
            category = category.lower().strip()
            valid_cats = ["tutor", "book", "solve", "generate", "schedule", "query"]
            return category if category in valid_cats else "query"
        except:
            return "query"

    async def route(self, prompt, file_path, user_id, convo_id):
        has_file = file_path is not None
        intent = await self.classify_intent(prompt, has_file)
        
        print(f" Intent Detected: '{intent}' (File: {has_file})")
        
      
        await self.history.save_message(user_id, convo_id, "user", prompt)
        history = await self.history.load_history(user_id, convo_id)

        response_text = ""
        pdf_output = None

        

        if intent == "book":
            if has_file:
                response_text = await self.book.process_book(file_path, prompt)
            else:
                response_text = " You asked to analyze a book, but didn't upload one. Please upload a PDF."

       
        elif intent in ["solve", "generate"]:
            if has_file:
                # This handles "Create a paper from this book" (generate) OR "Solve this image" (solve)
                result = await self.qp.process_file(file_path, mode=intent)
                response_text = result["text"]
                pdf_output = result["pdf_file"]
            elif intent == "generate":
                # Generate from RAG (No file)
                response_text = await self.rag.answer(f"Generate a question paper for: {prompt}", history, mode="query")
            else:
                response_text = " Please upload a file (PDF/Image) to solve."

        #SCHEDULE
        elif intent == "schedule":
            response_text = await self.sched.create_schedule(prompt)

        #TUTOR MODE
        elif intent == "tutor":
            response_text = await self.rag.answer(prompt, history, mode="tutor")

        #GENERAL QUERY (Default)
        else:
            response_text = await self.rag.answer(prompt, history, mode="query")

        # Save Assistant Response
        await self.history.save_message(user_id, convo_id, "assistant", response_text)
        
        return {"text": response_text, "pdf_file": pdf_output}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" Nexus Backend Starting...")
    global mongo_client, router
    

    if MONGO_URI:
        mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        chat_manager = ChatHistoryManager(mongo_client)
    else:
        print(" No Mongo URI. DB disabled.")
        chat_manager = None 

    router = RouterAgent(chat_manager)
    
    asyncio.create_task(router.rag.initialize_data())
    
    yield
    print("Nexus Backend Shutting Down...")
    if MONGO_URI: mongo_client.close()

app = FastAPI(lifespan=lifespan)

# CORS Setup
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://iiti-tutor-frontend.vercel.app",
    "https://iiti-tutor.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/route")
async def route_handler(
    request: Request,
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Unified Endpoint:
    - Accepts 'prompt' and optional 'file'.
    - Routes intelligently.
    - Returns JSON with 'text' and optional 'file_base64'.
    """
    temp_path = None
    
    if file:
        # Preserve extension for detection (pdf vs image)
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".pdf"
        
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # Manage Cookies
    user_id = request.cookies.get("user_id") or str(uuid.uuid4())
    convo_id = request.cookies.get("convo_id") or str(uuid.uuid4())

    try:
        result = await router.route(prompt, temp_path, user_id, convo_id)
        
        # Prepare Response
        resp_data = {"text": result["text"]}
        
        # Encode PDF if generated
        if result["pdf_file"]:
            result["pdf_file"].seek(0)
            b64_data = base64.b64encode(result["pdf_file"].read()).decode('utf-8')
            resp_data["file_base64"] = b64_data
            resp_data["mime_type"] = "application/pdf"
            resp_data["filename"] = "Nexus_Solution.pdf"

        # Return JSON
        response = JSONResponse(resp_data)
        response.set_cookie("user_id", user_id, secure=True, samesite="None")
        response.set_cookie("convo_id", convo_id, secure=True, samesite="None")
        return response

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse({"text": f"Server Error: {str(e)}"}, status_code=500)
    finally:
        # Cleanup Temp File
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def health_check():
    status = "ready" if router.rag.is_ready else "loading_data"
    return {"status": status, "msg": "Nexus AI is Live"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)