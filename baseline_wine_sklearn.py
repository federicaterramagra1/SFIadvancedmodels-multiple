# baseline_wine_sklearn.py
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def check_split_baseline(seed=42, test_frac=0.20):
    X, y = load_wine(return_X_y=True)
    # stesso split del tuo loader: prima train+val vs test
    X_trval, X_te, y_trval, y_te = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    sc = StandardScaler().fit(X_trval)   # fit SOLO su train+val
    X_trval = sc.transform(X_trval)
    X_te    = sc.transform(X_te)

    clf = LogisticRegression(max_iter=5000, multi_class="auto")
    clf.fit(X_trval, y_trval)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"[baseline sklearn] test_acc = {acc*100:.2f}%  (seed={seed}, test_frac={test_frac})")

if __name__ == "__main__":
    check_split_baseline()
