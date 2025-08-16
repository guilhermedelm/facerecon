import numpy as np
import embedding

def Add_new(person_name,new_image):
    global X,Y

    for img_path in new_image:
        emb = embedding.extract_embedding(img_path)
        if emb is not None:
            X = np.vstack([X,emb])
            Y = np.append(Y,person_name)