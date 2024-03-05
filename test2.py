import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
model = load_model('sign_language_prediction.h5')
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand con


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                             )
    

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

actions=['afternoon', 'asl', 'baby', 'bad', 'bathroom', 'blue', 'book',
       'boy', 'brother', 'bye', 'chat', 'cheese', 'children', 'class',
       'cook', 'cookies', 'day', 'deaf', 'dog', "don't want", 'drink',
       'eat-food', 'father', 'fine', 'finish', 'forget', 'girl', 'go',
       'good', 'grandfather', 'grandmother', 'hamburger', 'happy',
       'hello', 'help', 'home', 'hour', 'husband', 'kitchen', 'know',
       'like', 'living room', 'look, see, watch', 'man', 'milk', 'minute',
       'month', 'more', 'mother', 'need', 'night', 'no', 'noon', 'not',
       'not yet', 'orange', 'pay attention', 'pink', 'please', 'right',
       'sad', 'sentence', 'signing', 'so-so', 'student', 'thank-you',
       'time', 'want', 'week', 'when', 'where', 'who', 'wife', 'woman',
       'wrong', 'year', 'yes', 'you', 'your']

from scipy import stats
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color_index = num % len(colors)  # Get index within color list length
    return output_frame
predicted_actions=None
def index_page():
    st.title("Welcome to Signers")
    image_url = "Screenshot 2024-03-04 131800.png"
    st.write("Here You can know about sign languages from live video,it is a Deep learning Model and only about 100 gestures included.Go to the sidebar and select 'Go and test' to live see")
    # Display the image
    st.image(image_url, use_column_width=True)
def display_video():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        st.title("test")
        stop_button_pressed = st.button("Stop")
        ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

        if ctx.video_transformer:
            frame = ctx.video_transformer.out_image
            results = mediapipe_detection(frame, model)
            keypoints = extract_keypoints(results)

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-7:]
            
            if len(sequence) == 7:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_actions=actions[np.argmax(res)]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
                #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            encoded_frame = cv2.imencode('.jpg', image)[1].tobytes()
            frame_placeholder.image(encoded_frame, channels="RGB")
def main():
    page = st.sidebar.selectbox("Go to", ["Index Page", "Go and test"])

    if page == "Index Page":
        index_page()
    elif page == "Go and test":
        display_video()
if __name__ == "__main__":
    main()
