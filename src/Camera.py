import cv2
import torch
import sys
import numpy as np
import os
#from matplotlib import pyplot as plt
import time
from collections import OrderedDict

from Resnet2plus1d import r2plus1d_18

#import mediapipe as mp

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
model = r2plus1d_18(pretrained=True, num_classes=226)
#model.load_state_dict(torch.load('D:/Projects/SignLanguageTranslator/checkpoint/rgb_final_finetuned.pth'))
    # load pretrained
checkpoint = torch.load('D:/Projects/SignLanguageTranslator/checkpoint/rgb_final_finetuned.pth')
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:] # remove 'module.'
    new_state_dict[name]=v
model.load_state_dict(new_state_dict)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model.cuda()
model = model.to(device)



# def mediapipe_detection(image, modelx):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = modelx.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# def draw_styled_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
#                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
#                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                              ) 
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                              ) 
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                              ) 
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
#                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                              )
def preprocess_frame(frame):
    # Resize the frame to a specific size
    frame = cv2.resize(frame, (45,45))

    # Convert the frame to a numpy array
    frame = np.array(frame)

    # Normalize the frame
    frame = frame / 255.0

    # Add an additional dimension to the frame (since the model expects a 4D tensor as input)
    frame = np.expand_dims(frame, axis=0)
    
    return frame
def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])
def process_predictions(predicted_class_index, frame):
    # Extract the predicted class from the predictions

    

    #predicted_class = torch.max(predictions)


    # Add the predicted class to the frame
    cv2.putText(frame, str(predicted_class_index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
def plot_boxes(self, results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        bgr = (0, 255, 0) # color of the box
        classes = self.model.names # Get the name of label index
        label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
        cv2.rectangle(frame, \
                      (x1, y1), (x2, y2), \
                       bgr, 2) #Plot the boxes
        cv2.putText(frame,\
                    classes[labels[i]], \
                    (x1, y1), \
                    label_font, 0.9, bgr, 2) #Put a label over box.
        return frame
cap = cv2.VideoCapture(0)
while cap.isOpened():

        ret, frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #input_data = preprocess_frame(frame)
        image=cv2.resize(frame,(45,45),interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        input_tensor = torch.tensor(image).float()
        input_tensor = input_tensor.reshape(1, 3, 45, 45, 1)
        #input_tensor = input_tensor.reshape(1, 3, 45, 45, 1)
        #input_tensor = input_tensor.permute(2, 4, 3, 0, 1)
        input_tensor = input_tensor.to('cuda')

        

        predictions = model(input_tensor)

        predicted_class_index = predictions.argmax().item()

        #if predictions.max().item() > 0:
        process_predictions(predicted_class_index, frame)

        cv2.imshow('Frame', frame)
        #image, results = mediapipe_detection(frame, holistic)
        #print(image)
        

        #draw_styled_landmarks(image, results)

        #cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():

#         ret, frame = cap.read()

#         input_data = preprocess_frame(frame)


#         input_tensor = torch.tensor(input_data).float()


#         predictions = model(input_tensor)
    

#         process_predictions(predictions, frame)


#         cv2.imshow('Frame', frame)
#         #image, results = mediapipe_detection(frame, holistic)
#         #print(image)
        

#         #draw_styled_landmarks(image, results)

#         #cv2.imshow('OpenCV Feed', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()


#draw_landmarks(frame, results)
#plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))