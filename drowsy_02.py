import cv2
import mediapipe as mp
from shapely.geometry import Polygon
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


eye = {'left': 
       {'coordinates': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
        'values': {'x': [], 'y': []}},
       'right': 
       {'coordinates': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
        'values': {'x': [], 'y': []}}}
iris = {'left': 
       {'coordinates': [473, 474, 475, 476, 477],
        'values': {'x': [], 'y': []}},
       'right': 
       {'coordinates': [468, 469, 470, 471, 472],
        'values': {'x': [], 'y': []}}}

left_corners = [362, 263]
right_corners = [33, 133]

radius = 2
color = (0, 255, 0)
thickness = 1

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

		# Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
						image=image,
						landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
      face_landmarks = results.multi_face_landmarks[0]
      

      eye['left']['values']['x'] = [] 
      eye['left']['values']['y'] = [] 
      eye['right']['values']['x'] = [] 
      eye['right']['values']['y'] = [] 
      for each_point in eye['left']['coordinates']:
        eye['left']['values']['x'].append(face_landmarks.landmark[each_point].x)
        eye['left']['values']['y'].append(face_landmarks.landmark[each_point].y)
      for each_point in eye['right']['coordinates']:
        eye['right']['values']['x'].append(face_landmarks.landmark[each_point].x)
        eye['right']['values']['y'].append(face_landmarks.landmark[each_point].y)

        
      
    eye_left_area = Polygon(zip(eye['left']['values']['x'],eye['left']['values']['y'])).area
    eye_right_area = Polygon(zip(eye['right']['values']['x'],eye['right']['values']['y'])).area

    distance_left = dist = np.sqrt(np.sum(np.square(
      np.array((face_landmarks.landmark[left_corners[0]].x, face_landmarks.landmark[left_corners[0]].y, face_landmarks.landmark[left_corners[0]].z))-
      np.array((face_landmarks.landmark[left_corners[1]].x, face_landmarks.landmark[left_corners[1]].y, face_landmarks.landmark[left_corners[1]].z))
      )))
    distance_right = dist = np.sqrt(np.sum(np.square(
      np.array((face_landmarks.landmark[right_corners[0]].x, face_landmarks.landmark[right_corners[0]].y, face_landmarks.landmark[right_corners[0]].z))-
      np.array((face_landmarks.landmark[right_corners[1]].x, face_landmarks.landmark[right_corners[1]].y, face_landmarks.landmark[right_corners[1]].z))
      )))
    
    eye_left_area_norm = eye_left_area / distance_left
    eye_right_area_norm = eye_right_area / distance_right

    print(f'Left Eye Area: {eye_left_area_norm}, Right Eye Area: {eye_right_area_norm}')
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))   
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()