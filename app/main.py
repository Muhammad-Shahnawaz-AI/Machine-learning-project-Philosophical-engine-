from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
from app.scraper import scrape_philosophical_texts
from app.training import train_models
from app.models import PhilosophicalEngine
from app.visualizer import PhilosophicalEngineGUI

app = FastAPI(title="Philosophical Engine")
engine = PhilosophicalEngine()
scraper_task = None

class PredictionRequest(BaseModel):
    text: str
    model: str

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

@app.get("/modes")
def list_modes():
    return {"modes": engine.available_modes()}

@app.post("/predict")
def predict(request: PredictionRequest):
    return {"message": "Use the GUI for pure NumPy model interaction."}

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    return {"message": "Feedback functionality is disabled for the pure engine implementation."}

if __name__ == "__main__":
    PhilosophicalEngineGUI(engine).run()
