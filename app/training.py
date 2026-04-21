import json
import os
from app.models import PhilosophicalEngine

def train_models(engine: PhilosophicalEngine):
    data_dir = 'app/data'
    quotes_file = os.path.join(data_dir, 'quotes.json')
    
    if not os.path.exists(quotes_file):
        return
    
    with open(quotes_file, 'r') as f:
        quotes = json.load(f)
    
    # Dummy labels for supervised learning (e.g., philosophical schools)
    labels = [i % 3 for i in range(len(quotes))]  # 0: Ancient, 1: Modern, 2: Contemporary
    
    engine.train_supervised(quotes, labels)
    engine.train_unsupervised(quotes)