import os
import json
import re
import datetime
import asyncio
import io
import shutil
import tempfile
import uuid
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
from contextlib import asynccontextmanager

import httpx
import motor.motor_asyncio
from fastapi import FastAPI, File, UploadFile, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import faiss
import numpy as np
from datetime import datetime, timezone
from huggingface_hub import InferenceClient


def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    print(f"[MEMORY {tag}] Used: {process.memory_info().rss / 1024**2:.2f} MB")


class ChatHistoryManager:
    def __init__(self, mongo_client):
        self.client = mongo_client
        self.db = self.client["IITI_Tutor_DB"]
        self.collection = self.db["chat_history"]

    async def save_message(self, user_id: str, conversation_id: str, role: str, message_content: str):
        message = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": message_content,
            "timestamp": datetime.now(timezone.utc)
        }
        try:
            await self.collection.insert_one(message)
        except Exception as e:
            print(f"Database save error: {str(e)}")

    async def load_history(self, user_id: str, conversation_id: str, limit: int = 5) -> list[dict]:
        try:
            cursor = self.collection.find(
                {"user_id": user_id, "conversation_id": conversation_id},
                projection={"role": 1, "content": 1, "_id": 0}
            ).sort("timestamp", -1).limit(limit)
            history = [doc async for doc in cursor]
            return history[::-1]
        except Exception as e:
            print(f"Error loading history: {e}")
            return []

class QueryBot:
    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.MODEL_NAME = "llama-3.1-8b-instant"
        self.GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
        self.EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
        self.hf_client = InferenceClient(token=self.HF_TOKEN)
        
        self.chunks = []
        self.metadata = []
        self.index = None
        self.is_ready = False  

    async def initialize_data(self):
        """Loads data in background"""
        try:
            print("[Background] Loading Course Data...")
            self.chunks, self.metadata = await self.load_course_chunks()
            
            if self.chunks:
                print(f"[Background] Generating Embeddings for {len(self.chunks)} chunks...")
                loop = asyncio.get_event_loop()
                embeddings_list = []
                
                batch_size = 20
                for i in range(0, len(self.chunks), batch_size):
                    batch = self.chunks[i:i+batch_size]
                    try:
                        task = loop.run_in_executor(None, lambda b=batch: self.hf_client.feature_extraction(b, model=self.EMBEDDING_MODEL_ID))
                        batch_resp = await task
                        
                        batch_arr = np.array(batch_resp)
                        if len(batch_arr.shape) == 3:
                            batch_emb = np.mean(batch_arr, axis=1)
                        else:
                            batch_emb = batch_arr
                        embeddings_list.append(batch_emb)
                    except Exception as e:
                        print(f"Batch error: {e}")

                if embeddings_list:
                    final_embeddings = np.concatenate(embeddings_list, axis=0)
                    self.index = faiss.IndexFlatL2(final_embeddings.shape[1])
                    self.index.add(final_embeddings)
                    self.is_ready = True
                    print(f"[Background] FAISS Index Ready.")
                else:
                    print("No embeddings generated.")
            else:
                print("No course data found.")
                self.is_ready = True 
        except Exception as e:
            print(f" Background Initialization Failed: {e}")
            self.is_ready = True 

    async def load_course_chunks(self):
        try:
            db = mongo_client["IITI_Tutor_DB"]
            collection = db["First_Year_Curriculum"]
            cursor = collection.find({})
            raw_courses = await cursor.to_list(length=None)
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            raw_courses = []

        chunks = []
        metadata = []

        for course in raw_courses:
            code = course.get("Course Code", "")
            title = course.get("Course Title", course.get("Title", ""))
            full_text = f"{code}\n{title}\n"
            for k, v in course.items():
                if k == "_id": continue
                if isinstance(v, (list, dict)):
                    full_text += f"\n{k}: {json.dumps(v)}"
                else:
                    full_text += f"\n{k}: {v}"

            for paragraph in full_text.split("\n\n"):
                if len(paragraph.strip()) > 50:
                    chunks.append(paragraph.strip())
                    metadata.append({"course": code, "title": title})

        return chunks, metadata

    async def retrieve_relevant_chunks(self, query, top_k=4, chat_history=None):
        if not self.is_ready:
            return []
        if not self.index:
            return []
            
        try:
            loop = asyncio.get_event_loop()
            query_emb = await loop.run_in_executor(
                None, 
                lambda: self.hf_client.feature_extraction(query, model=self.EMBEDDING_MODEL_ID)
            )
            query_vec = np.array(query_emb)
            if len(query_vec.shape) > 1:
                query_vec = np.mean(query_vec, axis=0)
            query_vec = query_vec.reshape(1, -1)
            
            D, I = self.index.search(query_vec, top_k)
            return [(self.chunks[i], self.metadata[i]) for i in I[0] if i < len(self.chunks)]
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    async def query_llama(self, query, context_chunks, chat_history: list = None):
        # Fallback if system is still loading
        if not self.is_ready:
             return {"text": "System is still warming up (loading course data). Please try again in 30 seconds.", "pdf_file": None}

        context = "\n\n".join(f"Chunk: {chunk}" for chunk, meta in context_chunks)
        messages = [{"role": "system", "content": "You are a helpful academic assistant."}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": f"{context}\n\nQuestion: {query}"})

        payload = {
            "model": self.MODEL_NAME,
            "messages": messages,
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.GROQ_API_KEY}", "Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            response = await client.post(self.GROQ_API_URL, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            return {"text": response.json()["choices"][0]["message"]["content"].strip(), "pdf_file": None}

class QuestionPaperBot:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")

    async def pdf_to_images(self, pdf_path, temp_dir):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            images = await loop.run_in_executor(pool, convert_from_path, pdf_path, 300)
            image_paths = []
            for i, page in enumerate(images):
                path = os.path.join(temp_dir, f"Page_{i+1}.jpg")
                page.save(path, "JPEG")
                image_paths.append(path)
            return image_paths

    async def extract_text(self, image_paths):
        loop = asyncio.get_event_loop()
        async def process_image(path):
            return await loop.run_in_executor(None, lambda: pytesseract.image_to_string(Image.open(path)))
        tasks = [process_image(path) for path in image_paths]
        texts = await asyncio.gather(*tasks)
        return "\n".join(texts)

    async def generate_text_via_llm(self, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

    def text_to_formatted_pdf(self, text):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 50
        lines = text.split('\n')
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
            if len(line) > 90:
                chunks = [line[i:i+90] for i in range(0, len(line), 90)]
                for chunk in chunks:
                    c.drawString(50, y, chunk)
                    y -= 14
            else:
                c.drawString(50, y, line)
                y -= 14
        c.save()
        buffer.seek(0)
        return {"text": text, "pdf_file": buffer}

    async def process_paper(self, pdf_path, mode="answer"):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = await self.pdf_to_images(pdf_path, temp_dir)
            raw_text = await self.extract_text(image_paths)
            if mode == "answer":
                prompt = f"Provide solutions for:\n{raw_text}"
            else:
                prompt = f"Generate a similar question paper based on:\n{raw_text}"
            generated_text = await self.generate_text_via_llm(prompt)
            return self.text_to_formatted_pdf(generated_text)

class Scheduler:
    def __init__(self):
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.MODEL_NAME = "llama-3.1-8b-instant"
        self.GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    async def run_scheduler(self, initial_prompt: str):
        prompt = f"Create a daily schedule for: {initial_prompt}"
        headers = {"Authorization": f"Bearer {self.GROQ_API_KEY}"}
        payload = {
            "model": self.MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.GROQ_API_URL, headers=headers, json=payload, timeout=30)
            return {"text": response.json()["choices"][0]["message"]["content"].strip(), "pdf_file": None}

class RouterAgent:
    def __init__(self, chat_history_manager):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.chat_history_manager = chat_history_manager

    async def classify_prompt(self, user_prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        prompt = f"""Classify into one: questionpaper, scheduler, or query.
        User Input: {user_prompt}
        Output (one word only):"""
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                content = response.json()["choices"][0]["message"]["content"].strip().lower()
                if "question" in content: return "questionpaper"
                if "sched" in content: return "scheduler"
                return "query"
        except:
            return "query"

    async def route(self, user_prompt: str, user_id: str, conversation_id: str, file=None):
        query_type = await self.classify_prompt(user_prompt)
        await self.chat_history_manager.save_message(user_id, conversation_id, "user", user_prompt)
        chat_history = await self.chat_history_manager.load_history(user_id, conversation_id)

        result = None
        if query_type == "questionpaper":
            qp_bot = QuestionPaperBot()
            mode = "generate" if "generate" in user_prompt.lower() else "answer"
            result = await qp_bot.process_paper(file, mode=mode)
        elif query_type == "scheduler":
            sch_bot = Scheduler()
            result = await sch_bot.run_scheduler(user_prompt)
        else:
            relevant_chunks = await global_query_bot.retrieve_relevant_chunks(user_prompt, chat_history=chat_history)
            result = await global_query_bot.query_llama(user_prompt, relevant_chunks, chat_history=chat_history)

        if result and result.get("text"):
            await self.chat_history_manager.save_message(user_id, conversation_id, "assistant", result["text"])
        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application Startup")
    
    global mongo_client, chat_history_manager, global_query_bot, router_agent
    
    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGO_URI"))
    chat_history_manager = ChatHistoryManager(mongo_client)
    
    global_query_bot = QueryBot()
    router_agent = RouterAgent(chat_history_manager)
    
    # Schedule data loading in background so startup doesn't timeout
    asyncio.create_task(global_query_bot.initialize_data())
    
    yield
    print("Application Shutdown")
    mongo_client.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/route")
async def route_handler(request: Request, response: Response, prompt: str = Form(...), file: Optional[UploadFile] = File(None)):
    temp_file_path = None
    if file:
        fd, temp_file_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    current_user_id = request.cookies.get("user_id") or str(uuid.uuid4())
    current_conversation_id = request.cookies.get("convo_id") or str(uuid.uuid4())

    try:
        result = await router_agent.route(prompt, current_user_id, current_conversation_id, temp_file_path)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    text = result.get("text", "")
    pdf_file = result.get("pdf_file", None)

    if pdf_file:
        pdf_file.seek(0)
        return StreamingResponse(pdf_file, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=response.pdf"})

    resp = JSONResponse(content={"text": text})
    resp.set_cookie(key="user_id", value=current_user_id, samesite="None", secure=True)
    resp.set_cookie(key="convo_id", value=current_conversation_id, samesite="None", secure=True)
    return resp

@app.get("/")
async def health_check():
    status = "ready" if global_query_bot.is_ready else "initializing"
    return {"status": status, "message": "IITI Tutor Backend is Live"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)