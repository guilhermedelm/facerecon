import mediapipe as mp
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog
from deepface import DeepFace
import tensorflow as tf



mp_face=mp.solutions.face_detection

face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
def extract_embedding(img):
  #img = cv2.imread(img_path)
  img_rgb = rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  results = face_detection.process(img_rgb)
      
    
  if results.detections:
    print("Tipo da variável results:", type(results))
    print("Conteúdo de results:", results)
    bbox = results.detections[0].location_data.relative_bounding_box

    h,w,_ = img_rgb.shape
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)

    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
      return None

    face = cv2.resize(face, (160,160))

    embedding = DeepFace.represent(img_path = face, model_name = "Facenet")[0]["embedding"]

    return embedding



def load_dataset():
    embeddings = []
    labels = []

    datasetdir = "/content/files"

    for person in os.listdir(datasetdir):
        person_path = os.path.join(datasetdir,person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path,img_name)
            emb = extract_embedding(img_path)

            if emb is not None:
                embeddings.append(emb)
                labels.append(person)
def save_embedding(X,Y,cache_file="faces_embeddings.npz"):
  if os.path.exists("faces_embeddings.npz"):
    np.savez(cache_file,X=X,Y=Y)

  else:
    try:
        # Tenta criar um novo arquivo 'novo_arquivo.txt'
        with open('faces_embeddings.npz', 'x') as arquivo:
            arquivo.write('Conteúdo do novo arquivo.')
        print("Arquivo 'faces_embeddings.npz' criado com sucesso.")
    except FileExistsError:
        print("O arquivo 'novo_arquivo.txt' já existe.")

def load_embeddings(cache_file="faces_embeddings.npz"):
  
  if os.path.exists(cache_file):
    data = np.load(cache_file, allow_pickle=True)
    return data["X"], data["y"]
  return [], []


def capture_image():
  video = cv2.VideoCapture(0)
  captured_image = None
  while True:
    ret,frame = video.read()
    if not ret:
      print("erro ret")
      break

    cv2.imshow("Pressione ESPAÇO para capturar, ESC para sair", frame)

    key =cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC -> cancelar
      break
    elif key == 32:  # SPACE -> captura
      captured_image = frame.copy()
      break
  video.release()
  print("foto tirada")
  cv2.destroyAllWindows()
  return captured_image
    

def ask_name():
    root = tk.Tk()
    root.withdraw()  # esconde a janela principal
    name = simpledialog.askstring("Novo Cadastro", "Digite o nome da pessoa:")
    root.destroy()
    if name == "":
       print("nome inválido")
       return
    return name


def Add_new():
    global X,Y

    img = capture_image()
    print("passou")
    if img is None:
       print("nenhuma imagem capturada")
       return
    
    '''img = cv2.imread(img_path)
    img_rgb = rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
      results = face_detection.process(img_rgb)'''



    emb = extract_embedding(img)
    if emb is not None:

      person_name = ask_name()

      if len(X) == 0:
        X = np.array([emb])
        Y = np.array([person_name])
      else:
        X = np.vstack([X,emb])
        Y = np.append(Y,person_name)
      save_embedding(X,Y)
      print(f"Pessoa {person_name} adicionada com sucesso!")


    else:
      print("Erro ao gerar embedding. Nenhum rosto detectado.")




    


