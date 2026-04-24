from app.models import PhilosophicalEngine


def train_models(engine: PhilosophicalEngine):
    classification_X, classification_y = engine.generate_dataset("classification", n_samples=300, random_state=42)
    regression_X, regression_y = engine.generate_dataset("regression", n_samples=250, random_state=42)

    history = {}
    for mode in engine.available_modes():
        if mode == "Teleology":
            history[mode] = engine.train_model(mode, regression_X, regression_y, epochs=180, learning_rate=0.05)
        elif mode == "Taxonomy of Being":
            history[mode] = engine.train_model(mode, classification_X)
        else:
            history[mode] = engine.train_model(mode, classification_X, classification_y, epochs=120, learning_rate=0.05)
    return history
