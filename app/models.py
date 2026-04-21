import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_is_fitted

class PhilosophicalEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.supervised_models = {
            'svm': SVC(kernel='linear'),
            'decision_tree': DecisionTreeClassifier(),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }
        self.unsupervised_model = KMeans(n_clusters=5, random_state=42)
        self.q_table = {}  # For Q-Learning
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.models_dir = 'app/models'
        os.makedirs(self.models_dir, exist_ok=True)

    def preprocess_data(self, texts, labels=None):
        X = self.vectorizer.fit_transform(texts) if labels is not None else self.vectorizer.transform(texts)
        return X, labels

    def train_supervised(self, texts, labels):
        X, y = self.preprocess_data(texts, labels)
        for name, model in self.supervised_models.items():
            model.fit(X, y)
            self.save_model(model, f'{name}_model.pkl')

    def predict_supervised(self, text, model_name):
        X = self.preprocess_data([text])
        model = self.load_model(f'{model_name}_model.pkl')
        return model.predict(X)[0]

    def train_unsupervised(self, texts):
        X, _ = self.preprocess_data(texts)
        self.unsupervised_model.fit(X)
        self.save_model(self.unsupervised_model, 'kmeans_model.pkl')

    def predict_unsupervised(self, text):
        X = self.preprocess_data([text])
        return self.unsupervised_model.predict(X)[0]

    def q_learning_step(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in range(10)}  # Assume 10 possible actions (quotes)
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in range(10)}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def suggest_quote(self, current_quote_index):
        if current_quote_index not in self.q_table:
            return np.random.randint(0, 10)  # Explore
        else:
            return max(self.q_table[current_quote_index], key=self.q_table[current_quote_index].get)

    def save_model(self, model, filename):
        with open(os.path.join(self.models_dir, filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        with open(os.path.join(self.models_dir, filename), 'rb') as f:
            return pickle.load(f)