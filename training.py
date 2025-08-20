from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, Y):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, Y)
    return knn