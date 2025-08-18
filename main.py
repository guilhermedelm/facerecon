import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import embedding
import training as tr

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["embeddings"], data["labels"]   

knn = tr.train_knn(X,y)
