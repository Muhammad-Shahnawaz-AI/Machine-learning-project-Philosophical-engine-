curl -X POST "http://localhost:8000/predict_supervised" \
  -H "Content-Type: application/json" \
  -d '{"text": "The unexamined life is not worth living", "model": "svm"}'from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
from app.scraper import scrape_philosophical_texts
from app.training import train_models
from app.models import PhilosophicalEngine

app = FastAPI(title="Philosophical Engine")

engine = PhilosophicalEngine()
scraper_task = None

class PredictionRequest(BaseModel):
    text: str
    model: str  # 'svm', 'decision_tree', 'xgboost'

class FeedbackRequest(BaseModel):
    current_quote_index: int
    action: int
    reward: float
    next_quote_index: int

@app.post("/start_scraper")
async def start_scraper(background_tasks: BackgroundTasks):
    global scraper_task
    if scraper_task is None or scraper_task.done():
        scraper_task = asyncio.create_task(scrape_philosophical_texts())
        background_tasks.add_task(scrape_philosophical_texts)
        return {"message": "Scraper started"}
    return {"message": "Scraper already running"}

@app.post("/stop_scraper")
async def stop_scraper():
    global scraper_task
    if scraper_task and not scraper_task.done():
        scraper_task.cancel()
        return {"message": "Scraper stopped"}
    return {"message": "Scraper not running"}

@app.post("/train")
async def train():
    train_models(engine)
    return {"message": "Training completed"}

@app.post("/predict_supervised")
async def predict_supervised(request: PredictionRequest):
    prediction = engine.predict_supervised(request.text, request.model)
    return {"prediction": prediction}

@app.post("/predict_unsupervised")
async def predict_unsupervised(text: str):
    cluster = engine.predict_unsupervised(text)
    return {"cluster": cluster}

@app.post("/suggest_quote")
async def suggest_quote(current_quote_index: int):
    suggestion = engine.suggest_quote(current_quote_index)
    return {"suggested_quote_index": suggestion}

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    engine.q_learning_step(request.current_quote_index, request.action, request.reward, request.next_quote_index)
    return {"message": "Feedback received"}