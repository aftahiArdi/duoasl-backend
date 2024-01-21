
# !pip3 install tensorflow opencv-python mediapipe scikit-learn matplotlib

import cv2 
import numpy as np 
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 




def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion
    # cv2Color converts images from one colorspace to another
    image.flags.writeable = False
    results = model.process(image) # make prediction from image grame
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color conversion
    return image, results


def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) # draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) # draw face connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) # draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))



def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose, face, lh, rh])


# extract_keypoints(results)
# path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_Data")

#Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# thirty videos with of data
no_sequences = 30

# videos are going to be 30 frames of length
sequence_length = 30



for action in actions:
    # for each action
    for sequence in range(no_sequences):
        try:
            # if folder already exist, will pass, else make numbered folders in action
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass



from sklearn.model_selection import train_test_split # used for training and testing
from tensorflow.keras.utils import to_categorical # used to make labels

label_map = {label:num for num, label in enumerate(actions)}
label_map
sequences, labels = [], []

# sequences are feature data (X), labels are target data (y)
for action in actions:
    for sequence in range(no_sequences):
        window = [] # all frames for specific sequence (video)
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            # loads numpy frame 0, frame 1, .. 
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        # append 
X = np.array(sequences) # makes to np array
y = to_categorical(labels).astype(int) # uses one hot encoding to prevent bias

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05) # splits data
from tensorflow.keras.models import Sequential # Sequential lets you build a sequential NN
from tensorflow.keras.layers import LSTM, Dense # LSTM is temporal (involves time) and lets build model
from tensorflow.keras.callbacks import TensorBoard # allows to logging in tensor board
# Tensorboard is webapp to see neural network training

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential() # easy to make neural network
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.load_weights('gen6.h5')

model.summary()

actions[np.argmax(y_test[3])]

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist() # conv
yhat = np.argmax(yhat, axis =1).tolist()


# code to access openCV

cap = cv2.VideoCapture("../uploads/video.mp4")  # Access video file # access video cam on device port 0

# with is used to handle resource management
# set mediapipe model

# min detection is initial detection, tracking confidence is preceding tracking confidence
# if you want higher inital confidence in answer, then increase it and vice versa
action_list = []
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
	while cap.isOpened(): # while camera is on

		# Read feed (reading frame from webcam)
		ret, frame = cap.read()

		if ret == False:
			break

		

		# make detections
		image, results = mediapipe_detection(frame, holistic)
		
		# print(results)

		# draw landmarks
		draw_styled_landmarks(image, results)
		
        # 2. prediction logic
		keypoints = extract_keypoints(results)
		sequence.insert(0,keypoints)
		sequence = sequence[:30]
		
		if (len(sequence) == 30):
			res = model.predict(np.expand_dims(sequence, axis=0))[0]
			# expand dims allows us to test one sequence since its expecting (0, 30, 1662)
			print("Seen action: {}", actions[np.argmax(res)])
			action_list.append(actions[np.argmax(res)])


		# show frame to screen
		# cv2.imshow('OpenCV Feed', image)
		
		# break gracefully
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
print("test")
OUTPUT_PATH = os.path.join("outputs") 
output = os.path.join(OUTPUT_PATH, "test")
print(output)
np.save(output, action_list)

# release cv2 and close all windows
cv2.destroyAllWindows()
cv2.waitKey(1)
cap.release()
