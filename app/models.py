import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if y is None:
            raise ValueError("Ground-truth labels are required for scoring.")
        preds = self.predict(X)
        if self._is_classification(y):
            return float(np.mean(preds == y))
        return float(np.mean((preds - y) ** 2))

    def _is_classification(self, y: np.ndarray) -> bool:
        return y.dtype.kind in "iu" or np.all(np.equal(np.mod(y, 1), 0))


class VerdictEngine:
    INSIGHTS = {
        "Relativism": "Each decision is shaped by its nearest neighbors, reminding us that perspective defines truth.",
        "Socratic Method": "A tree of questions grows until the simplest answer reveals the structure of reason.",
        "Epistemological Doubt": "Every probability curve is a meditation on evidence and uncertainty.",
        "Teleology": "A line drawn toward purpose shows how past data bends toward future meaning.",
        "Taxonomy of Being": "Clusters emerge as categories of existence; similarity becomes the taxonomy of the soul.",
        "Collective Consciousness": "A forest of minds votes together, demonstrating how strength arises from plural understanding.",
        "Hegelian Dialectics": "Residuals are synthesized through thesis and antithesis until a higher truth emerges.",
        "The Neural Nexus": "A layered network learns hidden patterns, echoing a neural architecture of thought."
    }

    @classmethod
    def explain(cls, mode: str, history: Dict[str, Any], X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> str:
        base = cls.INSIGHTS.get(mode, "The engine searches for meaning through algorithmic reflection.")
        if history is None:
            return base
        summary = [base]
        if "loss" in history:
            loss = history["loss"][-1]
            summary.append(f"Final convergence loss: {loss:.4f}.")
            if loss < 0.05:
                summary.append("The dialectic has reached a deep harmony.")
            elif loss < 0.25:
                summary.append("Meaningful structure is emerging from the noise.")
            else:
                summary.append("The inquiry remains open, inviting further refinement.")
        elif "accuracy" in history:
            accuracy = history["accuracy"][-1]
            summary.append(f"Final accuracy: {accuracy * 100:.1f}%.")
            if accuracy > 0.9:
                summary.append("The algorithmic mind has aligned closely with the observed world.")
            elif accuracy > 0.7:
                summary.append("The model grasps a compelling but incomplete truth.")
            else:
                summary.append("The search for knowledge continues through more questions and evidence.")
        elif mode == "Taxonomy of Being" and X is not None:
            summary.append("Clusters show how the same data can belong to different forms of being.")
        return " ".join(summary)


class DatasetGenerator:
    @staticmethod
    def classification(n_samples: int = 300, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        centers = np.array([[-3.0, -3.0], [2.5, 2.0], [-1.5, 3.0]])
        points_per_cluster = n_samples // len(centers)
        X = []
        y = []
        for label, center in enumerate(centers):
            samples = rng.normal(loc=center, scale=1.0, size=(points_per_cluster, 2))
            X.append(samples)
            y.append(np.full(points_per_cluster, label, dtype=int))
        X = np.vstack(X)
        y = np.concatenate(y)
        shuffle = rng.permutation(len(X))
        return X[shuffle], y[shuffle]

    @staticmethod
    def regression(n_samples: int = 250, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        X = rng.uniform(-4.0, 4.0, size=(n_samples, 2))
        y = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 2.0 * np.sin(0.7 * X[:, 0]) + rng.normal(scale=1.4, size=n_samples)
        return X, y


class KNNRelativism(BaseModel):
    def __init__(self, k: int = 5):
        super().__init__("Relativism")
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.X_train = X
        self.y_train = y
        self.fitted = True
        return {"accuracy": [self.score(X, y)]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        distances = np.linalg.norm(X[:, None, :] - self.X_train[None, :, :], axis=2)
        nearest = np.argsort(distances, axis=1)[:, : self.k]
        return np.array([np.bincount(self.y_train[row]).argmax() for row in nearest])


class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeSocratic(BaseModel):
    def __init__(self, max_depth: int = 6, min_samples_split: int = 3):
        super().__init__("Socratic Method")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.is_classification = True

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.is_classification = kwargs.get("is_classification", self._is_classification(y))
        self.root = self._grow_tree(X, y, depth=0)
        self.fitted = True
        return {"accuracy": [self.score(X, y)]} if self.is_classification else {"loss": [self.score(X, y)]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionNode:
        if len(y) < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            return DecisionNode(value=self._leaf_value(y))
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return DecisionNode(value=self._leaf_value(y))
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionNode(feature_index=feature_idx, threshold=threshold, left=left, right=right)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = -np.inf
        best_idx, best_thresh = None, None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = feature_index
                    best_thresh = threshold
        return best_idx, best_thresh

    def _information_gain(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        if self.is_classification:
            return self._gini(parent) - (weight_left * self._gini(left) + weight_right * self._gini(right))
        return self._variance(parent) - (weight_left * self._variance(left) + weight_right * self._variance(right))

    def _gini(self, y: np.ndarray) -> float:
        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)
        return 1.0 - np.sum(probabilities**2)

    def _variance(self, y: np.ndarray) -> float:
        return float(np.var(y))

    def _leaf_value(self, y: np.ndarray):
        if self.is_classification:
            return int(np.bincount(y).argmax())
        return float(np.mean(y))

    def _traverse_tree(self, x: np.ndarray, node: DecisionNode):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class NaiveBayesEpistemologicalDoubt(BaseModel):
    def __init__(self):
        super().__init__("Epistemological Doubt")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.classes = np.unique(y)
        self.parameters = {}
        self.class_prior = {}
        for cls in self.classes:
            features = X[y == cls]
            self.parameters[cls] = {
                "mean": np.mean(features, axis=0),
                "var": np.var(features, axis=0) + 1e-9,
            }
            self.class_prior[cls] = len(features) / len(X)
        self.fitted = True
        return {"accuracy": [self.score(X, y)]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        return np.array([self._predict_row(x) for x in X])

    def _predict_row(self, x: np.ndarray) -> int:
        scores = []
        for cls in self.classes:
            mean = self.parameters[cls]["mean"]
            var = self.parameters[cls]["var"]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var)
            scores.append(np.log(self.class_prior[cls]) + log_likelihood)
        return int(self.classes[np.argmax(scores)])


class LinearRegressionTeleology(BaseModel):
    def __init__(self, learning_rate: float = 0.05, epochs: int = 150):
        super().__init__("Teleology")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        lr = kwargs.get("learning_rate", self.learning_rate)
        epochs = kwargs.get("epochs", self.epochs)
        n_samples, n_features = X.shape
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        self.weights = np.zeros(n_features + 1)
        history = []
        for _ in range(epochs):
            predictions = X_b.dot(self.weights)
            error = predictions - y
            gradient = (2 / n_samples) * X_b.T.dot(error)
            self.weights -= lr * gradient
            history.append(float(np.mean(error ** 2)))
        self.fitted = True
        return {"loss": history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b.dot(self.weights)


class KMeansTaxonomy(BaseModel):
    def __init__(self, n_clusters: int = 3, max_iter: int = 60):
        super().__init__("Taxonomy of Being")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        rng = np.random.default_rng(kwargs.get("random_state", 42))
        n_samples = X.shape[0]
        centroids = X[rng.choice(n_samples, self.n_clusters, replace=False)].copy()
        history = []
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(self.n_clusters)])
            inertia = float(np.mean(np.min(distances, axis=1)))
            history.append(inertia)
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        self.centroids = centroids
        self.fitted = True
        return {"loss": history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


class RandomForestCollectiveConsciousness(BaseModel):
    def __init__(self, n_estimators: int = 10, max_depth: int = 6, sample_rate: float = 0.8):
        super().__init__("Collective Consciousness")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_rate = sample_rate
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        n_samples = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, int(n_samples * self.sample_rate), replace=True)
            tree = DecisionTreeSocratic(max_depth=self.max_depth)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        self.fitted = True
        return {"accuracy": [self.score(X, y)]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        votes = np.vstack([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(votes[:, i]).argmax() for i in range(votes.shape[1])])


class GradientBoostingHegelianDialectics(BaseModel):
    def __init__(self, n_estimators: int = 10, learning_rate: float = 0.2):
        super().__init__("Hegelian Dialectics")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.init_value = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        self.trees = []
        self.init_value = float(np.mean(y))
        current = np.full_like(y, self.init_value, dtype=float)
        history = []
        for _ in range(self.n_estimators):
            residual = y - current
            stump = DecisionTreeSocratic(max_depth=1)
            stump.fit(X, residual, is_classification=False)
            update = stump.predict(X)
            current += self.learning_rate * update
            self.trees.append(stump)
            history.append(float(np.mean((y - current) ** 2)))
        self.fitted = True
        return {"loss": history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        output = np.full(X.shape[0], self.init_value, dtype=float)
        for tree in self.trees:
            output += self.learning_rate * tree.predict(X)
        return output


class NeuralNexus(BaseModel):
    def __init__(self, hidden_size: int = 16, learning_rate: float = 0.05, epochs: int = 180):
        super().__init__("The Neural Nexus")
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = {}
        self.is_classification = True

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        lr = kwargs.get("learning_rate", self.learning_rate)
        epochs = kwargs.get("epochs", self.epochs)
        self.is_classification = self._is_classification(y)
        n_samples, n_features = X.shape
        X = X.astype(float)
        y_target = y.reshape(-1, 1).astype(float)
        self.weights = {
            "W1": np.random.normal(scale=0.1, size=(n_features, self.hidden_size)),
            "b1": np.zeros(self.hidden_size),
            "W2": np.random.normal(scale=0.1, size=(self.hidden_size, 1)),
            "b2": np.zeros(1),
        }
        history = []
        for _ in range(epochs):
            z1 = X.dot(self.weights["W1"]) + self.weights["b1"]
            a1 = np.tanh(z1)
            z2 = a1.dot(self.weights["W2"]) + self.weights["b2"]
            a2 = self._sigmoid(z2) if self.is_classification else z2
            error = a2 - y_target
            dW2 = a1.T.dot(error) / n_samples
            db2 = np.mean(error, axis=0)
            dA1 = error.dot(self.weights["W2"].T)
            dZ1 = dA1 * (1 - np.tanh(z1) ** 2)
            dW1 = X.T.dot(dZ1) / n_samples
            db1 = np.mean(dZ1, axis=0)
            self.weights["W2"] -= lr * dW2
            self.weights["b2"] -= lr * db2
            self.weights["W1"] -= lr * dW1
            self.weights["b1"] -= lr * db1
            history.append(float(np.mean(error ** 2)))
        self.fitted = True
        return {"loss": history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        z1 = X.dot(self.weights["W1"]) + self.weights["b1"]
        a1 = np.tanh(z1)
        z2 = a1.dot(self.weights["W2"]) + self.weights["b2"]
        a2 = self._sigmoid(z2) if self.is_classification else z2
        if self.is_classification:
            return (a2.flatten() >= 0.5).astype(int)
        return a2.flatten()


class PhilosophicalEngine:
    def __init__(self):
        self.modes = {
            "Relativism": KNNRelativism(k=5),
            "Socratic Method": DecisionTreeSocratic(max_depth=6),
            "Epistemological Doubt": NaiveBayesEpistemologicalDoubt(),
            "Teleology": LinearRegressionTeleology(learning_rate=0.05, epochs=150),
            "Taxonomy of Being": KMeansTaxonomy(n_clusters=3),
            "Collective Consciousness": RandomForestCollectiveConsciousness(n_estimators=10, max_depth=6),
            "Hegelian Dialectics": GradientBoostingHegelianDialectics(n_estimators=12, learning_rate=0.2),
            "The Neural Nexus": NeuralNexus(hidden_size=16, learning_rate=0.05, epochs=180),
        }

    def available_modes(self):
        return list(self.modes.keys())

    def generate_dataset(self, kind: str = "classification", n_samples: int = 300, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if kind == "regression":
            return DatasetGenerator.regression(n_samples=n_samples, random_state=random_state)
        return DatasetGenerator.classification(n_samples=n_samples, random_state=random_state)

    def train_model(self, mode: str, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        if mode not in self.modes:
            raise ValueError(f"Mode '{mode}' is not available.")
        if mode == "Taxonomy of Being":
            return self.modes[mode].fit(X, None, **kwargs)
        return self.modes[mode].fit(X, y, **kwargs)

    def infer(self, mode: str, X: np.ndarray) -> np.ndarray:
        if mode not in self.modes:
            raise ValueError(f"Mode '{mode}' is not available.")
        return self.modes[mode].predict(X)

    def verdict(self, mode: str, history: Dict[str, Any], X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> str:
        return VerdictEngine.explain(mode, history, X, y)
