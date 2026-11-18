# Sistema de asistencia por reconocimiento facial

Proyecto de **transfer learning** que registra la asistencia a clase reconociendo el rostro de los estudiantes con una cámara web. Usa el modelo preentrenado **dlib ResNet** (`dlib_face_recognition_resnet_model_v1`) como extractor de características: no se reentrena la red, sino que cada rostro se convierte en un vector de 128 números y se compara con los de la base de datos.

Cuando el sistema reconoce a un estudiante, muestra su nombre en pantalla y guarda en un archivo CSV su **código, nombre, programa, fecha y hora** de entrada (una sola vez por día).

## Cómo funciona

1. **Codificación:** `crear_encodings.py` recorre las fotos de cada estudiante, detecta el rostro y calcula su vector de 128 valores con el modelo ResNet de dlib. Los vectores se guardan en `encodings.pkl`.
2. **Reconocimiento:** `reconocer.py` abre la cámara, calcula el vector de cada rostro detectado y lo compara con la base de datos usando la distancia euclidiana. Si la distancia es menor al umbral, la persona se considera reconocida.
3. **Verificación:** para evitar errores, la persona debe ser reconocida en varias revisiones seguidas antes de registrarse.
4. **Registro:** la asistencia se guarda en `asistencia.csv` y, al cerrar el programa, se exporta a `asistencia.xlsx`.

## Estructura del proyecto

```
Reconocimiento_Facial/
├── fotos/                      # fotos de los estudiantes (NO se sube al repositorio)
│   ├── NombreEstudiante1/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── NombreEstudiante2/
├── modelos/
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   └── shape_predictor_5_face_landmarks.dat
├── descargar_modelos.py        # descarga los modelos de dlib
├── crear_encodings.py          # genera la base de datos de rostros
├── reconocer.py                # reconocimiento en vivo y registro de asistencia
├── probar.py                   # prueba rápida del entorno
├── estudiantes.csv             # datos de cada estudiante
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3 y una cámara web.
- Las librerías de `requirements.txt` (dlib, OpenCV, NumPy, pandas, openpyxl).
- Los dos modelos de dlib en la carpeta `modelos/`.

## Instalación

```bash
git clone https://github.com/alix1907/Reconocimiento_Facial.git
cd Reconocimiento_Facial

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Si `pip install dlib` falla en Windows porque pide un compilador de C++, instala la versión precompilada:

```bash
pip install dlib-bin
```

### Modelos

Si la carpeta `modelos/` no contiene los dos archivos `.dat`, descárgalos con:

```bash
python descargar_modelos.py
```

## Preparar tus datos

### 1. Fotos

Crea la carpeta `fotos/` y, dentro, **una subcarpeta por persona** con 3 a 5 fotos suyas (de frente, buena luz y con leves cambios de ángulo). En cada foto debe aparecer **una sola cara**; las fotos con varios rostros se omiten.

### 2. Archivo `estudiantes.csv`

Debe tener estas columnas, y la columna `carpeta` debe coincidir **exactamente** con el nombre de la subcarpeta en `fotos/`:

```csv
carpeta,codigo,nombre,programa
Ana,20231001,Ana Pérez,Ingeniería de Software
Luis,20231002,Luis Gómez,Ingeniería de Software
```

## Uso

**1. Generar la base de datos de rostros** (cada vez que agregues o cambies fotos):

```bash
python crear_encodings.py
```

**2. Iniciar el reconocimiento:**

```bash
python reconocer.py
```

- Se abre la cámara y se muestra el nombre sobre cada rostro reconocido.
- El panel de la derecha muestra quiénes ya registraron asistencia y a qué hora.
- Presiona **`q`** para salir. Al cerrar se genera `asistencia.xlsx` y se imprime un resumen de presentes y ausentes.

## Resultado

La asistencia queda en `asistencia.csv` (y `asistencia.xlsx`) con este formato:

| codigo | nombre | programa | fecha | hora |
|---|---|---|---|---|
| 20231001 | Ana Pérez | Ingeniería de Software | 2026-11-18 | 08:02:15 |

Cada persona se registra **una sola vez por día**. Si abres `asistencia.xlsx` mientras el programa corre, ciérralo antes de salir con `q`, porque Windows bloquea el archivo.

## Parámetros ajustables

Están al inicio de `reconocer.py`:

| Parámetro | Qué hace | Cuándo cambiarlo |
|---|---|---|
| `UMBRAL` (0.6) | Distancia máxima para aceptar un rostro | Bájalo (0.5) si confunde personas; súbelo (0.65) si no reconoce |
| `CONFIRMACIONES` (3) | Revisiones seguidas antes de registrar | Súbelo si registra por error |
| `CADA_N_FRAMES` (3) | Procesa 1 de cada N fotogramas | Súbelo si va lento |
| `ESCALA` (0.5) | Reduce la imagen para detectar más rápido | Bájala si va lento |

## Privacidad

Las fotos son datos biométricos y personales, por eso la carpeta `fotos/`, `encodings.pkl` y los archivos de asistencia están en el `.gitignore` y **no se suben al repositorio**.

## Limitaciones

- El reconocimiento depende de la luz, el ángulo y la calidad de las fotos de referencia.
- No distingue fotos impresas o pantallas de rostros reales (no tiene detección de vida).
- Funciona mejor con pocos rostros a la vez frente a la cámara.

## Créditos

- Modelos de reconocimiento facial: [dlib](http://dlib.net) de Davis King (repositorio [dlib-models](https://github.com/davisking/dlib-models); revisa allí su licencia).
- Proyecto académico del curso [Matemáticas y Simulación], [Universidad Surcolombiana].

**Integrantes:** [Alix Zoraya Fonseca Mendez], [Danna Sofia Andrade Quimbaya], [Juan David Escobar]
