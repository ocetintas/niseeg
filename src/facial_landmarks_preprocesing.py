import dlib
import numpy as np
import cv2
import os
import os.path as osp
import shutil

# PARAMETERS TO CHANGE FOR PROCESSING
DATA_PATH = 'data/face_videos'
subjects = [10, 12, 13]
OUTPUT_PATH = 'data/landmarks'


predictor_path = 'models/shape_predictor_81_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


for s in subjects:
    print("Processing subject: ", s)
    subj_code = 's' + '{:02d}'.format(s)
    landmark_path = osp.join(OUTPUT_PATH, subj_code)

    # Create the folder
    if osp.exists(landmark_path):
        print("Found existing landmarks. Deleting them and replacing them for new ones")
        shutil.rmtree(landmark_path)
    os.makedirs(landmark_path)

    for t in range(1, 41):
        print("     Processing trial: ", t)
        # Get the file
        trial_code = 'trial' + '{:02d}'.format(t)
        seq_file = osp.join(DATA_PATH, subj_code, subj_code+'_'+trial_code+'.avi')

        cap = cv2.VideoCapture(seq_file)  # Get the frames
        sequence_landmarks = []  # Array to store the landmarks
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                dets = detector(frame, 0)
                if not dets:
                    print("Face detection is missed, using zero array")
                    landmarks = np.zeros((81, 2))
                    sequence_landmarks.append(landmarks)
                else:
                    for k, d in enumerate(dets):
                        shape = predictor(frame, d)
                        landmarks = np.array([[p.x/frame.shape[1], p.y/frame.shape[0]] for p in shape.parts()])  # Normalize
                        sequence_landmarks.append(landmarks)
            else:
                # Sequence is over
                break

        sequence_landmarks = np.array(sequence_landmarks)
        landmark_file = osp.join(landmark_path,  subj_code+'_'+trial_code+'.npy')
        np.save(landmark_file, sequence_landmarks)

