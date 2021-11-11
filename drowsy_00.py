import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
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
      left_eye = []
      right_eye = []
      left_iris = []
      right_iris = []
    
      # Center Coordinates
      point = 477
      center_coordinates = (round(face_landmarks.landmark[point].x * image.shape[1]), 
                            round(face_landmarks.landmark[point].y * image.shape[0]))
      # Radius of circle
      radius = 2
      # Blue color in BGR
      color = (0, 255, 0)
      # Line thickness of 2 px
      thickness = 1
      # Using cv2.circle() method
      # Draw a circle with blue line borders of thickness of 2 px
      image = cv2.circle(image, center_coordinates, radius, color, thickness)

    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))   
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()