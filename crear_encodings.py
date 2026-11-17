import dlib, cv2, os, pickle, numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("modelos/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("modelos/dlib_face_recognition_resnet_model_v1.dat")

encodings, carpetas = [], []
for carpeta in os.listdir("fotos"):
    ruta = os.path.join("fotos", carpeta)
    if not os.path.isdir(ruta):
        continue
    for foto in os.listdir(ruta):
        img = cv2.imread(os.path.join(ruta, foto))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        caras = detector(rgb, 1)
        if len(caras) != 1:
            print(f"Omitida {carpeta}/{foto}: {len(caras)} rostros detectados")
            continue
        vec = np.array(facerec.compute_face_descriptor(rgb, sp(rgb, caras[0])))
        encodings.append(vec)
        carpetas.append(carpeta)

with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "carpetas": carpetas}, f)
print(f"Listo: {len(encodings)} rostros codificados")