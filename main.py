import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
from playsound import playsound
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")


# calculate eye aspect ratio
def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b)/(2.0 * c)
    return ear


# threshold value
threshold = 0.25

# frame count
flag = 0

frame_check = 20

# eye landmarks of the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(grey, 0)
    for subject in subjects:
        shape_grey = predict(grey, subject)

        # converting shape to list of X,Y coordinates
        shape = face_utils.shape_to_np(shape_grey)

        leftEye = shape[lStart:lEnd]  # Corrected line
        rightEye = shape[rStart:rEnd]  # Corrected line

        # calculate right and left ear values
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar)/ 2.0

        # create green outline of the image
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < threshold:
            flag += 1
            print(flag)
            if flag <= frame_check:
                cv2.putText(frame, "**** ALERT ****", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "**** ALERT ****", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
