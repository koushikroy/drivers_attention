# Drivers Attention (Drowsiness Detection)

- In this project, our objective is to detect if the driver of a vehicle is attentive or drowsy. 
- We will be using the webcam video footage as the input source.
- And apply mediapipe facemesh in the input image frames and extract the face landmark positions. 
- We will calculate the area of the eye and depending on the area, we will decide if the eye is open or closed.
- We normalize the eye area by dividing the eye area by the iris area.
- If the driver's eye is closed for a predefined period of time, our model will aleart him.
