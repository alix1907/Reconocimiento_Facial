import dlib, cv2
print("dlib", dlib.__version__, "| OpenCV", cv2.__version__)
sp = dlib.shape_predictor("modelos/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("modelos/dlib_face_recognition_resnet_model_v1.dat")
print("Modelos cargados correctamente")