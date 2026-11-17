import urllib.request, bz2, os

os.makedirs("modelos", exist_ok=True)
archivos = [
    "dlib_face_recognition_resnet_model_v1.dat",
    "shape_predictor_5_face_landmarks.dat",
]
for nombre in archivos:
    url = f"http://dlib.net/files/{nombre}.bz2"
    print("Descargando", nombre)
    datos = urllib.request.urlopen(url).read()
    with open(f"modelos/{nombre}", "wb") as f:
        f.write(bz2.decompress(datos))
print("Listo")
