import cv2
import numpy as np
import face_recognition
import os

path = './app/images'
image_list = []
name_list = []

my_list = os.listdir(path)

for img in my_list:
    if os.path.splitext(img)[1].lower() in ['.jpg', '.png', '.jpeg']:
        cur_img = cv2.imread(os.path.join(path, img))
        image_list.append(cur_img)
        img_name = os.path.splitext(img)[0]
        name_list.append(img_name)

def find_encodings(img_list, img_name_list):
    signatures_db = []
    count = 1
    for my_img, name in zip(img_list, img_name_list):
        if my_img is None:
            print(f'Impossible de lire l\'image : {name}')
            continue
        img = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        if face_locations:
            signature = face_recognition.face_encodings(img, face_locations)[0]
            signature_class = signature.tolist() + [name]
            signatures_db.append(signature_class)
        else:
            print(f'Aucun visage détecté dans l\'image : {name}')
        print(f'{int((count / len(img_list)) * 100)}% extrait')
        count += 1

    signatures_db = np.array(signatures_db)
    np.save('./app/FaceSignatures_db.npy', signatures_db, allow_pickle=True)
    print('Base de données de signatures enregistrée')

def main():
    find_encodings(image_list, name_list)

if __name__ == '__main__':
    main()