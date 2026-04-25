# Philosophical Engine

A pure NumPy-based philosophical machine learning engine with interactive visualization and a symbolic verdict system.

## Overview

This repository implements a high-performance Philosophical Engine from scratch using only:
- `NumPy`
- `Matplotlib`
- `Tkinter` (Local GUI)
- `FastAPI` (Web GUI & REST API)
- `Jinja2` (Web Templates)
- `Chart.js` (Web Visualizations)

It avoids high-level ML libraries such as `scikit-learn` and `xgboost`, with manual implementations for:
- K-Nearest Neighbors
- Decision Trees
- Naive Bayes
- Linear Regression
- K-Means Clustering
- Random Forest
- Gradient Boosting
- A Multi-layer Neural Network with backpropagation

The app includes **dual interface options**:
1. **Local Tkinter GUI** - Desktop application with real-time visualizations
2. **Web-based GUI** - Modern browser interface with interactive charts and enhanced UX
3. **REST API** - Programmatic access for integration

Both GUIs visualize real-time training progress, decision boundaries, and philosophical verdicts derived from model convergence.

## Project Structure

```
Machine-learning-project-Philosophical-engine-/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI server with web GUI and API endpoints
│   ├── models.py          # Pure NumPy model implementations
│   ├── scraper.py         # Philosophical text scraper
│   ├── training.py        # Training utilities
│   ├── visualizer.py      # Tkinter GUI and plotting
│   ├── templates/         # Web GUI HTML templates
│   │   └── index.html     # Web-based GUI interface
│   ├── data/              # Data storage
│   └── models/            # Optional model persistence
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.10+
- Tkinter installed in your Python environment

### Local Setup

```bash
git clone https://github.com/Muhammad-Shahnawaz-AI/Machine-learning-project-Philosophical-engine-.git
cd Machine-learning-project-Philosophical-engine-
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

The project requires the following packages (automatically installed via requirements.txt):
- `fastapi==0.104.1` - Web framework for API and GUI
- `uvicorn[standard]==0.24.0` - ASGI server
- `numpy==1.26.2` - Numerical computations
- `matplotlib==3.8.4` - Plotting and visualization
- `beautifulsoup4==4.12.2` - Web scraping
- `requests==2.31.0` - HTTP client
- `jinja2==3.1.2` - HTML templating for web GUI
- `python-multipart==0.0.6` - Form data handling

## Usage

### Web-Based GUI (Recommended)

The modern web interface provides the best user experience with interactive charts and responsive design.

```bash
uvicorn app.main:app --reload
```

Access the web GUI at: `http://127.0.0.1:8000`

**Web GUI Features:**
- Modern, responsive design with gradient styling
- Interactive mode selection with all 8 philosophical modes
- Real-time data visualization using Chart.js
- Training progress with live metrics
- Decision boundary visualization
- Philosophical verdict display
- Scraper controls (start/stop)
- Mobile-friendly interface

### Local Desktop GUI

Traditional Tkinter desktop application for local use.

```bash
python app/main.py
```

**Local GUI Features:**
- Desktop-based interface
- Real-time training visualization
- Decision boundary plotting
- Philosophical verdict display
- All 8 modes available

### REST API

Programmatic access for integration and automation.

```bash
uvicorn app.main:app --reload
```

**API Endpoints:**
- `GET /` - Web GUI interface
- `GET /api` - API information and endpoints
- `GET /modes` - List all available philosophical modes
- `POST /train` - Train all models
- `POST /start_scraper` - Start the philosophical text scraper
- `POST /stop_scraper` - Stop the scraper
- `POST /predict` - Make predictions (GUI recommended)
- `POST /feedback` - Submit feedback (disabled in pure engine)

#### API Examples

```bash
# Get API info
curl http://127.0.0.1:8000/api

# List available modes
curl http://127.0.0.1:8000/modes

# Train models
curl -X POST http://127.0.0.1:8000/train

# Start scraper
curl -X POST http://127.0.0.1:8000/start_scraper
```

## Philosophical Modes of Inquiry

Each machine learning algorithm is expressed as a philosophical mode:

- **Relativism** — `KNNRelativism` (K-Nearest Neighbors)
- **Socratic Method** — `DecisionTreeSocratic` (Decision Tree)
- **Epistemological Doubt** — `NaiveBayesEpistemologicalDoubt` (Naive Bayes)
- **Teleology** — `LinearRegressionTeleology` (Linear Regression)
- **Taxonomy of Being** — `KMeansTaxonomy` (K-Means Clustering)
- **Collective Consciousness** — `RandomForestCollectiveConsciousness` (Random Forest)
- **Hegelian Dialectics** — `GradientBoostingHegelianDialectics` (Gradient Boosting)
- **The Neural Nexus** — `NeuralNexus` (Multi-layer Neural Network)

## Implementation Notes

- `app/models.py`: defines the abstract base model and every philosophical algorithm.
- `app/visualizer.py`: creates the interactive GUI with Matplotlib plots.
- `app/training.py`: validates training for all modes with synthetic datasets.
- `app/main.py`: serves as both the GUI launcher and a FastAPI compatibility layer.

## Dependencies

This project uses only:
- `fastapi`
- `uvicorn[standard]`
- `numpy`
- `matplotlib`
- `beautifulsoup4`
- `requests`

It no longer depends on `scikit-learn` or `xgboost`.

## Development

### Run GUI Application

```bash
python app/main.py
```

### Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

### Validate training

Use `app/training.py` or the GUI to exercise all modes and inspect the training curves.

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure you're running from the project root directory
2. **Matplotlib DLL Issues**: The project uses matplotlib==3.8.4 for compatibility
3. **Local GUI Not Showing**: Check that Tkinter is properly installed with your Python distribution
4. **Web GUI Not Loading**: Ensure FastAPI server is running and accessible at port 8000
5. **Template Not Found**: Verify `app/templates/index.html` exists and permissions are correct

### Windows Users

If you encounter path issues, make sure to run commands from the project root:
```cmd
cd c:\ML_Projects
uvicorn app.main:app --reload  # For Web GUI
python app/main.py              # For Local GUI
```

### Web GUI Issues

- **Blank Page**: Check browser console for JavaScript errors
- **API Not Responding**: Verify server is running on `http://127.0.0.1:8000`
- **Charts Not Loading**: Ensure Chart.js CDN is accessible (internet connection required)
- **CORS Errors**: Web GUI runs on same domain, should not encounter CORS issues

### Port Conflicts

If port 8000 is in use:
```bash
uvicorn app.main:app --port 8001 --reload
```
Then access at `http://127.0.0.1:8001`

## Notes

- This project is intended as an educational demonstration of manual algorithm implementation.
- **Three interface options available**: Web GUI (recommended), Local Desktop GUI, and REST API
- All major learning algorithms are implemented manually in NumPy for transparency and study.
- Web GUI requires internet connection for Chart.js CDN (first load only)
- The project has been thoroughly tested and validated for proper functionality.
- Web GUI provides enhanced user experience with modern design and interactive visualizations.
