# Philosophical Engine

A machine learning-powered application that analyzes philosophical texts, provides intelligent quote suggestions, and implements reinforcement learning for personalized recommendations.

## Course Information
Machine Learning Course, Semester 4

## Overview

The Philosophical Engine is an AI system that combines multiple machine learning techniques to process and analyze philosophical texts. It features:

- **Web scraping** for collecting philosophical quotes and texts
- **Supervised learning** models (SVM, Decision Tree, XGBoost) for text classification
- **Unsupervised learning** (K-Means clustering) for text grouping
- **Reinforcement learning** (Q-Learning) for intelligent quote recommendations
- **RESTful API** built with FastAPI for easy integration

## Features

- **Automated Data Collection**: Background web scraping from philosophical sources
- **Multi-Model Classification**: Choose between SVM, Decision Tree, or XGBoost for supervised learning
- **Text Clustering**: Unsupervised analysis to group similar philosophical concepts
- **Smart Recommendations**: Q-Learning algorithm that learns user preferences for quote suggestions
- **Real-time Feedback**: Continuous learning from user interactions
- **Docker Support**: Containerized deployment for easy scaling

## Project Structure

```
philosophical-engine/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and endpoints
│   ├── models.py        # ML models and algorithms
│   ├── scraper.py       # Web scraping functionality
│   ├── training.py      # Model training logic
│   ├── data/            # Scraped data storage
│   └── models/          # Trained model storage
├── Dockerfile           # Docker container configuration
├── docker-compose.yml   # Multi-container setup
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
```

## Installation

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Muhammad-Shahnawaz-AI/Machine-learning-project-Philosophical-engine-.git
   cd Machine-learning-project-Philosophical-engine-
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually:**
   ```bash
   docker build -t philosophical-engine .
   docker run -p 8000:8000 philosophical-engine
   ```

## Usage

### Running the Application

#### Local Development
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Docker
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

#### Core Endpoints

##### Data Collection
- **POST** `/start_scraper` - Start background scraping of philosophical texts
- **POST** `/stop_scraper` - Stop the background scraper

##### Model Training
- **POST** `/train` - Train all ML models using scraped data

##### Predictions
- **POST** `/predict_supervised`
  - Classify text using supervised models
  - Body: `{"text": "philosophical text", "model": "svm|decision_tree|xgboost"}`

- **POST** `/predict_unsupervised`
  - Cluster text using unsupervised learning
  - Body: `{"text": "philosophical text"}`

##### Recommendations
- **POST** `/suggest_quote`
  - Get intelligent quote suggestion
  - Body: `{"current_quote_index": 0}`

- **POST** `/feedback`
  - Provide feedback for reinforcement learning
  - Body: `{"current_quote_index": 0, "action": 1, "reward": 0.8, "next_quote_index": 2}`

### Example Workflow

1. **Start the scraper:**
   ```bash
   curl -X POST "http://localhost:8000/start_scraper"
   ```

2. **Wait for data collection** (check `app/data/quotes.json`)

3. **Train the models:**
   ```bash
   curl -X POST "http://localhost:8000/train"
   ```

4. **Make predictions:**
   ```bash
   curl -X POST "http://localhost:8000/predict_supervised" \
     -H "Content-Type: application/json" \
     -d '{"text": "The only thing I know is that I know nothing", "model": "svm"}'
   ```

5. **Get recommendations:**
   ```bash
   curl -X POST "http://localhost:8000/suggest_quote" \
     -H "Content-Type: application/json" \
     -d '{"current_quote_index": 0}'
   ```

## Machine Learning Models

### Supervised Learning
- **Support Vector Machine (SVM)**: Effective for high-dimensional text classification
- **Decision Tree**: Interpretable model for philosophical text categorization
- **XGBoost**: Gradient boosting for improved accuracy

### Unsupervised Learning
- **K-Means Clustering**: Groups philosophical texts into conceptual clusters

### Reinforcement Learning
- **Q-Learning**: Learns user preferences for quote recommendations
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation

## Data Processing

- **Text Vectorization**: TF-IDF transformation for numerical representation
- **Preprocessing**: Stop word removal and feature extraction
- **Model Persistence**: Automatic saving/loading of trained models

## Dependencies

Key libraries used:
- **FastAPI**: Modern web framework for API development
- **scikit-learn**: Machine learning algorithms and utilities
- **XGBoost**: Gradient boosting framework
- **BeautifulSoup4**: HTML parsing for web scraping
- **aiohttp**: Asynchronous HTTP client
- **NumPy/Pandas**: Data manipulation and analysis

## Configuration

### Environment Variables
- `PYTHONPATH`: Set to `/app` in Docker environment

### Model Parameters
- Vectorizer: max_features=1000, stop_words='english'
- K-Means: n_clusters=5, random_state=42
- Q-Learning: alpha=0.1, gamma=0.9, epsilon=0.1

## Error Handling

The application includes comprehensive error handling:
- Model not trained: Returns informative messages
- Invalid input: FastAPI automatic validation
- Network errors: Graceful degradation in scraping

## Development

### Code Quality
- Modular architecture with separation of concerns
- Type hints and docstrings
- Async/await for non-blocking operations

### Testing
```bash
# Run with reload for development
python -m uvicorn app.main:app --reload
```

### Docker Development
```bash
# Rebuild on changes
docker-compose up --build --force-recreate
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of an educational machine learning course.

## Contact

For questions or feedback, please open an issue in the repository.
