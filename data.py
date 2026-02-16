
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

def load_data():
    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = dataset.data
    labels = dataset.target
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, len(dataset.target_names)
