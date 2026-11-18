import dlib, cv2, pickle, os, time, unicodedata
import numpy as np, pandas as pd
from datetime import datetime

# ---------- Configuración ----------
UMBRAL = 0.6            # más bajo = más estricto
CONFIRMACIONES = 3      # revisiones seguidas antes de registrar
CADA_N_FRAMES = 3       # reconoce 1 de cada N fotogramas
ESCALA = 0.5            # reduce la imagen para detectar más rápido
SALIDA_CSV = "asistencia.csv"
SALIDA_XLSX = "asistencia.xlsx"

def sin_tildes(texto):
    
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()

# ---------- Modelos y datos ----------
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

faltan = set(db_car) - set(est.index)
if faltan:
    raise SystemExit(f"Estas carpetas no están en estudiantes.csv: {sorted(faltan)}")

# ---------- Registros de hoy ----------
hoy = datetime.now().strftime("%Y-%m-%d")
ya = set()          # códigos ya registrados hoy
registros = []      # (hora, nombre) para el panel
if os.path.exists(SALIDA_CSV):
    previo = pd.read_csv(SALIDA_CSV, dtype=str)
    delhoy = previo[previo["fecha"] == hoy]
    ya = set(delhoy["codigo"])
    registros = [(h[:5], n) for h, n in zip(delhoy["hora"], delhoy["nombre"])]

def registrar(carpeta):
    ahora = datetime.now()
    fila = pd.DataFrame([{
        "codigo": est.loc[carpeta, "codigo"],
        "nombre": est.loc[carpeta, "nombre"],
        "programa": est.loc[carpeta, "programa"],
        "fecha": hoy,
        "hora": ahora.strftime("%H:%M:%S"),
    }])
    fila.to_csv(SALIDA_CSV, mode="a", header=not os.path.exists(SALIDA_CSV),
                index=False, encoding="utf-8-sig")
    return ahora.strftime("%H:%M")

def dibujar_panel(frame):
    h, w = frame.shape[:2]
    ancho = 270
    capa = frame.copy()
    cv2.rectangle(capa, (w - ancho, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(capa, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Presentes: {len(registros)}/{len(est)}",
                (w - ancho + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, hoy, (w - ancho + 10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for k, (hora, nombre) in enumerate(registros[::-1][:14]):
        linea = f"{hora} {sin_tildes(nombre)}"[:30]
        cv2.putText(frame, linea, (w - ancho + 10, 82 + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def exportar_excel():
    if os.path.exists(SALIDA_CSV):
        try:
            pd.read_csv(SALIDA_CSV, dtype=str).to_excel(SALIDA_XLSX, index=False)
            print(f"Excel guardado: {SALIDA_XLSX}")
        except PermissionError:
            print("No se pudo guardar el Excel: cierra asistencia.xlsx y vuelve a ejecutar.")

# ---------- Cámara ----------
cap = cv2.VideoCapture(0)
n = 0
resultados = []      # lo último que se reconoció, se dibuja en TODOS los fotogramas
cuenta = {}          # confirmaciones seguidas por persona
mensaje, mensaje_hasta = "", 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1

    if n % CADA_N_FRAMES == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chico = cv2.resize(rgb, (0, 0), fx=ESCALA, fy=ESCALA)
        nuevos, vistos = [], set()

        for cara in detector(chico, 0):
            rect = dlib.rectangle(int(cara.left() / ESCALA), int(cara.top() / ESCALA),
                                  int(cara.right() / ESCALA), int(cara.bottom() / ESCALA))
            vec = np.array(facerec.compute_face_descriptor(rgb, sp(rgb, rect)))
            dist = np.linalg.norm(db_enc - vec, axis=1)
            i = dist.argmin()

            if dist[i] < UMBRAL:
                carpeta = db_car[i]
                nombre = est.loc[carpeta, "nombre"]
                codigo = est.loc[carpeta, "codigo"]
                vistos.add(carpeta)
                cuenta[carpeta] = cuenta.get(carpeta, 0) + 1

                if codigo in ya:
                    texto, color = f"{nombre} - ya registrado", (255, 200, 0)
                elif cuenta[carpeta] >= CONFIRMACIONES:
                    hora = registrar(carpeta)
                    ya.add(codigo)
                    registros.append((hora, nombre))
                    mensaje = f"Asistencia registrada: {nombre}"
                    mensaje_hasta = time.time() + 3
                    texto, color = f"{nombre} - REGISTRADO", (0, 255, 0)
                else:
                    texto, color = f"{nombre} (verificando...)", (0, 255, 255)
            else:
                texto, color = "Desconocido", (0, 0, 255)

            nuevos.append((rect.left(), rect.top(), rect.right(), rect.bottom(), texto, color))

        for c in cuenta:                 # si no se vio en esta revisión, reinicia el conteo
            if c not in vistos:
                cuenta[c] = 0
        resultados = nuevos

    # Dibujo en cada fotograma (así el nombre no parpadea)
    for izq, arr, der, aba, texto, color in resultados:
        cv2.rectangle(frame, (izq, arr), (der, aba), color, 2)
        cv2.putText(frame, sin_tildes(texto), (izq, max(arr - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if time.time() < mensaje_hasta:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 270, 40), (0, 140, 0), -1)
        cv2.putText(frame, sin_tildes(mensaje), (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    dibujar_panel(frame)
    cv2.imshow("Asistencia (q para salir)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- Al cerrar ----------
exportar_excel()
presentes = set(ya)
todos = set(est["codigo"])
print(f"\nResumen del {hoy}: {len(presentes)} presentes de {len(todos)}")
ausentes = est[est["codigo"].isin(todos - presentes)]["nombre"].tolist()
if ausentes:
    print("Faltaron:", ", ".join(ausentes))