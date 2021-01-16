import dlib
import numpy as np
import cv2

predictor_path = 'models/shape_predictor_81_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

sequence_landmarks = []

cap = cv2.VideoCapture('data/face_videos/s01/s01_trial01.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        dets = detector(frame, 0)
        if not dets:
            print("Face detection is missed, using zero array")
            landmarks = np.zeros((81, 2))
        else:
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                landmarks = np.array([[p.x/frame.shape[1], p.y/frame.shape[0]] for p in shape.parts()])  # Normalize
                for num in range(shape.num_parts):
                    cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0, 255, 0), -1)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("q pressed")
                    break
        sequence_landmarks.append(landmarks)
    else:
        # Sequence is over
        break

sequence_landmarks = np.array(sequence_landmarks)
np.save("test.npy", sequence_landmarks)

