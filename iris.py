import cv2
import mediapipe as mp
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

radius = 2
color1 = (0, 0, 255)
color2 = (0, 255, 0)
thickness = 2

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
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
    #   for face_landmarks in results.multi_face_landmarks:
    #     mp_drawing.draw_landmarks(
	# 					image=image,
	# 					landmark_list=face_landmarks,
    #         connections=mp_face_mesh.FACEMESH_TESSELATION,
    #         landmark_drawing_spec=None,
    #         connection_drawing_spec=mp_drawing_styles
    #         .get_default_face_mesh_tesselation_style())
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

      iris['left']['values']['x'] = [] 
      iris['left']['values']['y'] = [] 
      iris['right']['values']['x'] = [] 
      iris['right']['values']['y'] = [] 
      for each_point in iris['left']['coordinates']:
        iris['left']['values']['x'].append(face_landmarks.landmark[each_point].x)
        iris['left']['values']['y'].append(face_landmarks.landmark[each_point].y)
      for each_point in iris['right']['coordinates']:
        iris['right']['values']['x'].append(face_landmarks.landmark[each_point].x)
        iris['right']['values']['y'].append(face_landmarks.landmark[each_point].y)

      # Iris Center Coordinates
      iris_left = (round(face_landmarks.landmark[iris['left']['coordinates'][2]].x * image.shape[1]), 
                        round(face_landmarks.landmark[eye['left']['coordinates'][2]].y * image.shape[0]))
      iris_right = (round(face_landmarks.landmark[iris['right']['coordinates'][0]].x * image.shape[1]), 
                        round(face_landmarks.landmark[eye['right']['coordinates'][0]].y * image.shape[0]))
      image = cv2.circle(image, iris_left, radius, color1, thickness)
      image = cv2.circle(image, iris_right, radius, color1, thickness)
      
      # Center of the eye using center of mass eqn
      left_x = round(sum(eye['left']['values']['x'])/len(eye['left']['values']['x']) * image.shape[1])
      left_y = round(sum(eye['left']['values']['y'])/len(eye['left']['values']['y']) * image.shape[0])

      right_x = round(sum(eye['right']['values']['x'])/len(eye['right']['values']['x']) * image.shape[1])
      right_y = round(sum(eye['right']['values']['y'])/len(eye['right']['values']['y']) * image.shape[0])

      image = cv2.circle(image, (left_x, left_y), radius, color2, thickness)
      image = cv2.circle(image, (right_x, right_y), radius, color2, thickness)
        

    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()