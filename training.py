from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, y):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn