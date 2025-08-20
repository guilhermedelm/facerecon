import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import embedding as em
import training as tr
global X,Y
#with open("embeddings.pkl", "rb") as f:
#    data = pickle.load(f)
#X, y = data["embeddings"], data["labels"]   

#knn = tr.train_knn(X,y)

#em.Add_new()
em.run()




