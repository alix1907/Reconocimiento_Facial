import dlib, cv2, pickle, os
import numpy as np, pandas as pd
from datetime import datetime

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("modelos/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("modelos/dlib_face_recognition_resnet_model_v1.dat")

db = pickle.load(open("encodings.pkl", "rb"))
db_enc = np.array(db["encodings"])
db_car = db["carpetas"]
est = pd.read_csv("estudiantes.csv", dtype=str, sep=None,
                  engine="python", encoding="utf-8-sig")
est.columns = est.columns.str.strip()
est = est.apply(lambda c: c.str.strip())
est = est.set_index("carpeta")

# Avisa si alguna carpeta de las fotos no está en el CSV
faltan = set(db_car) - set(est.index)
if faltan:
    raise SystemExit(f"Estas carpetas no están en estudiantes.csv: {sorted(faltan)}")

UMBRAL = 0.65
SALIDA = "asistencia.csv"
hoy = datetime.now().strftime("%Y-%m-%d")

ya = set()
if os.path.exists(SALIDA):
    previo = pd.read_csv(SALIDA, dtype=str)
    ya = set(previo[previo["fecha"] == hoy]["codigo"])

def registrar(carpeta):
    ahora = datetime.now()
    fila = pd.DataFrame([{
        "codigo": est.loc[carpeta, "codigo"],
        "nombre": est.loc[carpeta, "nombre"],
        "programa": est.loc[carpeta, "programa"],
        "fecha": hoy,
        "hora": ahora.strftime("%H:%M:%S"),
    }])
    fila.to_csv(SALIDA, mode="a", header=not os.path.exists(SALIDA),
                index=False, encoding="utf-8-sig")

cap = cv2.VideoCapture(0)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
    if n % 2 == 0:                      # procesa 1 de cada 2 frames
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for cara in detector(rgb, 0):
            vec = np.array(facerec.compute_face_descriptor(rgb, sp(rgb, cara)))
            dist = np.linalg.norm(db_enc - vec, axis=1)
            i = dist.argmin()
            if dist[i] < UMBRAL:
                carpeta = db_car[i]
                nombre = est.loc[carpeta, "nombre"]
                codigo = est.loc[carpeta, "codigo"]
                if codigo not in ya:
                    registrar(carpeta)
                    ya.add(codigo)
            else:
                nombre = "Desconocido"
            cv2.rectangle(frame, (cara.left(), cara.top()),
                          (cara.right(), cara.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (cara.left(), cara.top() - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Asistencia (q para salir)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()