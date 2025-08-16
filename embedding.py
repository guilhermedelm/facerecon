import mediapipe as mp
import cv2
import numpy as np
import os
mp_face_mesh=mp.solutions.face_detection
def extract_embedding(img_path):
  img = cv2.imread(img_path)
  img_rgb = rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  with mp_face_mesh.FaceMesh(model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.procces(img_rgb)

    if results.detection:
      bbox = results.detections[0].location_data.relative_bounding_box

      h,w,_ = img.shape
      x1 = int(bbox.xmin * w)
      y1 = int(bbox.ymin * h)
      x2 = int((bbox.xmin + bbox.width) * w)
      y2 = int((bbox.ymin + bbox.height) * h)

      face = img_rgb[y1:y2, x1:x2]
      face = cv2.resize(face, (160,160))

      embedding = embedder.embeddings([face])[0]
      return embedding
    return None


def load_dataset():
    
    datasetdir = "/content/files"

    for person in os.listdir(datasetdir):
        person_path = os.path.join(datasetdir,person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path,img_name)
            emb = extract_embedding(img_path)

            if emb is not None:
                embeddings.append(emb)
                labels.append(person)

embeddings = []
labels = []


  


