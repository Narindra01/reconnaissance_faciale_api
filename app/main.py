from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import face_recognition
from PIL import Image
import io
import os

app = FastAPI()

# Charger la base de données de signatures faciales existante
#db_path = os.path.join(os.path.dirname(__file__), 'FaceSignatures_db.npy')
db_path = os.path.join(os.path.dirname(__file__), '..', 'FaceSignatures_db.npy')
signatures_class = np.load(db_path, allow_pickle=True)
X = signatures_class[:, 0:-1].astype('float')
Y = signatures_class[:, -1]

@app.post('/api/reconnaissance-faciale')
async def reconnaissance_faciale(image: UploadFile = File(...)):
    contents = await image.read()
    img_pil = Image.open(io.BytesIO(contents))
    img_np = np.array(img_pil)
    imgR = cv2.resize(img_np, (0, 0), None, 0.25, 0.25)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    facesCurrent = face_recognition.face_locations(imgR)
    encodesCurrent = face_recognition.face_encodings(imgR, facesCurrent)

    results = []
    for encodeFace, faceloc in zip(encodesCurrent, facesCurrent):
        matches = face_recognition.compare_faces(X, encodeFace)
        faceDis = face_recognition.face_distance(X, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = Y[matchIndex]
            results.append({'name': name, 'distance': faceDis[matchIndex]})
    
    return results

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)