from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import imutils
import numpy as np
from PIL import Image
from io import BytesIO
import numpy as np
import warnings
import cv2
import torch
import logging
import os
import glob
from PIL import Image
import facenet_pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from typing import Union, Dict
import PIL
import platform
import sqlite3

app = Flask(__name__)

# Tu código de detección de caras y funciones relacionadas aquí
# Configuración básica
warnings.filterwarnings('ignore')

logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.WARNING,
)

# Detectar si se dispone de GPU cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(F'Running on device: {device}')

# Configuración del detector MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# Cargar modelo de detección de emociones
emotionModel = load_model("modelAsencios.h5")

# Tipos de emociones del detector
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Ruta de la base de datos
DATABASE_PATH = 'mi_base_de_datos.db'

def predict_emotion(face):
    # Preprocesamiento del rostro para la detección de emociones
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    # Predicción de emociones
    pred = emotionModel.predict(face)
    return pred[0]

def detectar_caras(imagen: Union[PIL.Image.Image, np.ndarray],
                   detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                   keep_all: bool        = True,
                   min_face_size: int    = 20,
                   thresholds: list      = [0.6, 0.7, 0.7],
                   device: str           = None,
                   min_confidence: float = 0.5,
                   fix_bbox: bool        = True,
                   verbose               = False)-> np.ndarray:
    
    # Comprobaciones iniciales
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray, PIL.Image`. Recibido {type(imagen)}."
        )

    if detector is None:
        logging.info('Iniciando detector MTCC')
        detector = MTCNN(
                        keep_all      = keep_all,
                        min_face_size = min_face_size,
                        thresholds    = thresholds,
                        post_process  = False,
                        device        = device
                   )
        
    # Detección de caras
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32)
        
    bboxes, probs = detector.detect(imagen, landmarks=False)
    
    if bboxes is None:
        bboxes = np.array([])
        probs  = np.array([])
    else:
        # Se descartan caras con una probabilidad estimada inferior a `min_confidence`.
        bboxes = bboxes[probs > min_confidence]
        probs  = probs[probs > min_confidence]
        
    logging.info(f'Número total de caras detectadas: {len(bboxes)}')
    logging.info(f'Número final de caras seleccionadas: {len(bboxes)}')

    # Corregir bounding boxes
    #---------------------------------------------------------------------------
    # Si alguna de las esquinas de la bounding box está fuera de la imagen, se
    # corrigen para que no sobrepase los márgenes.
    if len(bboxes) > 0 and fix_bbox:       
        for i, bbox in enumerate(bboxes):
            if bbox[0] < 0:
                bboxes[i][0] = 0
            if bbox[1] < 0:
                bboxes[i][1] = 0
            if bbox[2] > imagen.shape[1]:
                bboxes[i][2] = imagen.shape[1]
            if bbox[3] > imagen.shape[0]:
                bboxes[i][3] = imagen.shape[0]

    # Información de proceso
    if verbose:
        print("----------------")
        print("Imagen escaneada")
        print("----------------")
        print(f"Caras detectadas: {len(bboxes)}")
        print(f"Correción bounding boxes: {ix_bbox}")
        print(f"Coordenadas bounding boxes: {bboxes}")
        print(f"Confianza bounding boxes:{probs} ")
        print("")
    
    return bboxes.astype(int)

def mostrar_bboxes_cv2(imagen: Union[PIL.Image.Image, np.ndarray],
                       bboxes: np.ndarray,
                       identidades: list=None,
                       device: str='window') -> None:

    # Comprobaciones iniciales
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray`, `PIL.Image`. Recibido {type(imagen)}."
        )
        
    if identidades is not None:
        if len(bboxes) != len(identidades):
            raise Exception(
                '`identidades` debe tener el mismo número de elementos que `bboxes`.'
            )
    else:
        identidades = [None] * len(bboxes)

    # Mostrar la imagen y superponer bounding boxes      
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32) / 255
    
    if len(bboxes) > 0:
        
        for i, bbox in enumerate(bboxes):
            
            if identidades[i] is not None:

                # ---------------------------------------------------
                (Xi, Yi, Xf, Yf) = bbox.astype(int)
                face = imagen[Yi:Yf, Xi:Xf]
                emotion_pred = predict_emotion(face)
                emotion_label = classes[np.argmax(emotion_pred)]
                label = f"{emotion_label}"

                # ---------------------------------------------------

                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (0, 255, 0),
                    thickness = 2
                )
                
                cv2.putText(
                    img       = imagen, 
                    text      = f"{identidades[i]} {label}",  
                    org       = (bbox[0], bbox[1]-10), 
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1e-3 * imagen.shape[0],
                    color     = (0,255,0),
                    thickness = 2
                )
            else:
                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (255, 0, 0),
                    thickness = 2
                )
        
    return imagen

def extraer_caras(imagen: Union[PIL.Image.Image, np.ndarray],
                  bboxes: np.ndarray,
                  output_img_size: Union[list, tuple, np.ndarray]=[160, 160]) -> None:

    # Comprobaciones iniciales
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser np.ndarray, PIL.Image. Recibido {type(imagen)}."
        )
        
    # Recorte de cara
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen)
        
    if len(bboxes) > 0:
        caras = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Redimensionamiento del recorte
            cara = imagen[y1:y2, x1:x2]
            cara = Image.fromarray(cara)
            cara = cara.resize(tuple(output_img_size))
            cara = np.array(cara)
            caras.append(cara)

            
            
    caras = np.stack(caras, axis=0)

    return caras

def calcular_embeddings(img_caras: np.ndarray, encoder=None,
                        device: str=None) -> np.ndarray: 

    # Comprobaciones iniciales
    if not isinstance(img_caras, np.ndarray):
        raise Exception(
            f"`img_caras` debe ser np.ndarray {type(img_caras)}."
        )
        
    if img_caras.ndim != 4:
        raise Exception(
            f"`img_caras` debe ser np.ndarray con dimensiones [nº caras, ancho, alto, 3]."
            f" Recibido {img_caras.ndim}."
        )
        
    if encoder is None:
        logging.info('Iniciando encoder InceptionResnetV1')
        encoder = InceptionResnetV1(
                        pretrained = 'vggface2',
                        classify   = False,
                        device     = device
                   ).eval()
        
    # Calculo de embedings
    # El InceptionResnetV1 modelo requiere que las dimensiones de entrada sean
    # [nº caras, 3, ancho, alto]
    caras = np.moveaxis(img_caras, -1, 1)
    caras = caras.astype(np.float32) / 255
    caras = torch.tensor(caras)
    embeddings = encoder.forward(caras).detach().cpu().numpy()
    embeddings = embeddings
    return embeddings

def identificar_caras(embeddings: np.ndarray,
                      dic_referencia: dict,
                      threshold_similaridad: float = 0.6) -> list:
    
    identidades = []

    for i in range(embeddings.shape[0]):
        # Se calcula la similitud con cada uno de los perfiles de referencia.
        similitudes = {}
        for key, value in dic_referencia.items():
            # Asegurarse de que ambos embeddings tengan las mismas dimensiones
            value_embedding = np.mean(value, axis=0) if len(value) > 0 else np.zeros_like(embeddings[i])
            similitudes[key] = 1 - cosine(embeddings[i].flatten(), value_embedding.flatten())
        
        # Se identifica la persona de mayor similitud.
        identidad = max(similitudes, key=similitudes.get)
        # Si la similitud < threshold_similaridad, se etiqueta como None
        if similitudes[identidad] < threshold_similaridad:
            identidad = None
        
        identidades.append(identidad)

    return identidades

def crear_tabla_referencias(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS referencias (
            identidad TEXT PRIMARY KEY,
            embeddings BLOB
        )
    ''')

def cargar_diccionario_desde_bd(ruta_bd):
    conn = sqlite3.connect(ruta_bd)
    cursor = conn.cursor()
    crear_tabla_referencias(cursor)
    cursor.execute('SELECT * FROM referencias')
    rows = cursor.fetchall()
    conn.close()

    dic_referencia = {}
    for row in rows:
        identidad, embeddings_bytes = row
        embeddings = np.frombuffer(embeddings_bytes, dtype=np.float32)
        dic_referencia[identidad] = embeddings

    return dic_referencia

def guardar_diccionario_en_bd(dic_referencia, ruta_bd):
    conn = sqlite3.connect(ruta_bd)
    cursor = conn.cursor()

    # Crear la tabla si no existe
    crear_tabla_referencias(cursor)

    for identidad, embeddings in dic_referencia.items():
        # Convertir el array de numpy a bytes
        embeddings_bytes = np.array(embeddings).tobytes()

        # Intentar recuperar la entrada existente en la base de datos
        cursor.execute('SELECT embeddings FROM referencias WHERE identidad=?', (identidad,))
        resultado = cursor.fetchone()

        if resultado is None:
            # Si no hay entrada para esta identidad, insertar una nueva
            cursor.execute('INSERT INTO referencias (identidad, embeddings) VALUES (?, ?)', (identidad, embeddings_bytes))
        else:
            # Si ya hay entradas, actualizar la lista existente
            embeddings_antiguos = np.frombuffer(resultado[0], dtype=np.float32)
            embeddings_nuevos = np.vstack([embeddings_antiguos, embeddings])
            cursor.execute('UPDATE referencias SET embeddings=? WHERE identidad=?', (embeddings_nuevos.tobytes(), identidad))

    conn.commit()
    conn.close()

def crear_diccionario_referencias(folder_path: str,
                                  detector: facenet_pytorch.models.mtcnn.MTCNN = None,
                                  min_face_size: int = 40,
                                  thresholds: list = [0.6, 0.7, 0.7],
                                  min_confidence: float = 0.9,
                                  encoder=None,
                                  device: str = None,
                                  verbose: bool = False) -> dict:

    # ... (código anterior)

    new_dic_referencia = {}
    folders = glob.glob(folder_path + "/*")

    for folder in folders:

        if platform.system() in ['Linux', 'Darwin']:
            identidad = folder.split("/")[-1]
        else:
            identidad = folder.split("\\")[-1]

        logging.info(f'Obteniendo embeddings de: {identidad}')
        embeddings = []
        # Se lista todas las imágenes .jpg .jpeg .tif .png
        path_imagenes = glob.glob(folder + "/*.jpg")
        path_imagenes.extend(glob.glob(folder + "/*.jpeg"))
        path_imagenes.extend(glob.glob(folder + "/*.tif"))
        path_imagenes.extend(glob.glob(folder + "/*.png"))
        logging.info(f'Total imágenes referencia: {len(path_imagenes)}')

        for path_imagen in path_imagenes:
            logging.info(f'Leyendo imagen: {path_imagen}')
            imagen = Image.open(path_imagen)
            # Si la imagen es RGBA se pasa a RGB
            if np.array(imagen).shape[2] == 4:
                imagen = np.array(imagen)[:, :, :3]
                imagen = Image.fromarray(imagen)

            bbox = detectar_caras(
                imagen,
                detector=detector,
                min_confidence=min_confidence,
                verbose=False
            )

            if len(bbox) > 1:
                logging.warning(
                    f'Más de 2 caras detectadas en la imagen: {path_imagen}. '
                    f'Se descarta la imagen del diccionario de referencia.'
                )
                continue

            if len(bbox) == 0:
                logging.warning(
                    f'No se han detectado caras en la imagen: {path_imagen}.'
                )
                continue

            cara = extraer_caras(imagen, bbox)
            embedding = calcular_embeddings(cara, encoder=encoder)
            embeddings.append(embedding)

        if verbose:
            print(f"Identidad: {identidad} --- Imágenes referencia: {len(embeddings)}")

        # Asegurarse de que todos los embeddings tengan la misma forma
        embeddings = np.array(embeddings)
        if embeddings.shape[0] > 0:
            # Ajustar las dimensiones de los embeddings
            embeddings = np.mean(embeddings, axis=0)
            new_dic_referencia[identidad] = embeddings

    return new_dic_referencia

def pipeline_deteccion_webcam(dic_referencia: dict,
                             output_device: str = 'window',
                             path_output_video: str=os.getcwd(),
                             detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                             keep_all: bool=True,
                             min_face_size: int=40,
                             thresholds: list=[0.6, 0.7, 0.7],
                             device: str=None,
                             min_confidence: float=0.5,
                             fix_bbox: bool=True,
                             output_img_size: Union[list, tuple, np.ndarray]=[160, 160],
                             encoder=None,
                             threshold_similaridad: float=0.5,
                             ax=None,
                             verbose=False)-> None:
    

    capture = cv2.VideoCapture(0)

    frame_exist = True

    while(frame_exist):
        frame_exist, frame = capture.read()

        if not frame_exist:
            capture.release()
            #cv2.destroyAllWindows()
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Llamada a detectar_caras con el detector MTCNN
        bboxes = detectar_caras(
                    imagen         = frame,
                    detector       = detector,
                    keep_all       = keep_all,
                    min_face_size  = min_face_size,
                    thresholds     = thresholds,
                    device         = device,
                    min_confidence = min_confidence,
                    fix_bbox       = fix_bbox
                  )

        if len(bboxes) == 0:

            logging.info('No se han detectado caras en la imagen.')
            #cv2.imshow(output_device, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_procesado = frame
            
                             
        else:
            # Obtener caras de la imagen
            caras = extraer_caras(
                    imagen = frame,
                    bboxes = bboxes
                )
            embeddings = calcular_embeddings(
                    img_caras = caras,
                    encoder   = encoder
                )
            identidades = identificar_caras(
                    embeddings     = embeddings,
                    dic_referencia = dic_referencias,
                    threshold_similaridad = threshold_similaridad
                )
            frame_procesado = mostrar_bboxes_cv2(
                    imagen      = frame,
                    bboxes      = bboxes,
                    identidades = identidades,
                    device = output_device
                )

        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2RGB))
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    capture.release()

@app.route('/registrar', methods=['GET', 'POST'])
def registrar_persona():
    if request.method == 'POST':
        nombre = request.form['nombre']
        imagen = request.files['imagen']

        if nombre and imagen:
            # Guardar la imagen en el directorio de referencia
            path_guardado = f'./images/imagenes_referencia_reconocimiento_facial/{nombre}'
            if not os.path.exists(path_guardado):
                os.makedirs(path_guardado)

            path_imagen = os.path.join(path_guardado, imagen.filename)
            imagen.save(path_imagen)

            # Crear un nuevo diccionario solo para la nueva persona
            dic_persona_nueva = crear_diccionario_referencias(
                folder_path=path_guardado,
                min_face_size=40,
                min_confidence=0.9,
                device=device,
                verbose=True
            )

            # Agregar o actualizar el diccionario global
            dic_referencias = cargar_diccionario_desde_bd(DATABASE_PATH)
            dic_referencias[nombre] = dic_persona_nueva.get(nombre, {})

            # Guardar el diccionario actualizado en la base de datos
            guardar_diccionario_en_bd(dic_referencias, DATABASE_PATH)

            return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/video_feed')
def video_feed():
    return Response(pipeline_deteccion_webcam(dic_referencia=dic_referencias,threshold_similaridad=0.4),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Intentar cargar el diccionario desde la base de datos
    dic_referencias = cargar_diccionario_desde_bd(DATABASE_PATH)

    # Si el diccionario está vacío, crear uno nuevo
    if not dic_referencias:
        dic_referencias = crear_diccionario_referencias(
            folder_path='./images/imagenes_referencia_reconocimiento_facial',
            min_face_size=40,
            min_confidence=0.9,
            device=device,
            verbose=True
        )

        # Guardar el diccionario recién creado en la base de datos
        guardar_diccionario_en_bd(dic_referencias, DATABASE_PATH)

    app.run(debug=True)