# Philosophical Engine

A pure NumPy-based philosophical machine learning engine with interactive visualization and a symbolic verdict system.

## Overview

This repository implements a high-performance Philosophical Engine from scratch using only:
- `NumPy`
- `Matplotlib`
- `Tkinter`

It avoids high-level ML libraries such as `scikit-learn` and `xgboost`, with manual implementations for:
- K-Nearest Neighbors
- Decision Trees
- Naive Bayes
- Linear Regression
- K-Means Clustering
- Random Forest
- Gradient Boosting
- A Multi-layer Neural Network with backpropagation

The app includes a GUI that visualizes real-time training progress, decision boundaries, and philosophical verdicts derived from model convergence.

## Project Structure

```
Machine-learning-project-Philosophical-engine-/
├── app/
│   ├── __init__.py
│   ├── main.py            # Launcher for GUI and FastAPI compatibility
│   ├── models.py          # Pure NumPy model implementations
│   ├── scraper.py         # Legacy philosophical text scraper
│   ├── training.py        # Validation training utilities
│   ├── visualizer.py      # Tkinter GUI and plotting
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
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Launch the GUI

The recommended interface is the Tkinter GUI.

```bash
python -m app.main
```

The GUI supports:
- selecting a philosophical mode
- generating synthetic classification or regression datasets
- training models with live epoch curves
- visualizing decision boundaries for 2D data
- displaying a philosophical verdict based on convergence

### Legacy API

A FastAPI endpoint is still available for compatibility.

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Check the available modes at:
- `http://localhost:8000/modes`

> Note: The current implementation emphasizes the GUI and pure NumPy engine. Legacy REST endpoints are available but not the primary user experience.

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

### Run locally

```bash
python -m app.main
```

### Run the API

```bash
python -m uvicorn app.main:app --reload
```

### Validate training

Use `app/training.py` or the GUI to exercise all modes and inspect the training curves.

## Notes

- This project is intended as an educational demonstration of manual algorithm implementation.
- The GUI is the preferred frontend for this codebase.
- All major learning algorithms are implemented manually in NumPy for transparency and study.
