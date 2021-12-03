from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.layers import CuDNNLSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
import keras.backend as K
import keras_tuner
import pydotplus
import matplotlib.pyplot as plt
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import itertools, random
import os, copy, glob
import time, datetime
import argparse, easydict
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_number", required=True, type=int, default=1, help="0: Lite, 1: Full, 2: Heavy")
parser.add_argument("-i", "--input_path", required=True, type=str, help="INPUT 비디오를 지정해야 함")
parser.add_argument("-c", "--confidence", required=False, type=float, default=0.5, help="최소 신뢰도를 지정할 수 있음")
args = parser.parse_args()

model_number = args.model_number
input_path = args.input_path
confidence = args.confidence

ALL_LABEL_LIST = ["stand", "walk", "run", "sit", "greet_up_hands", "", "falldown", "cross_arms_stand", "cross_arms_sit",
             "", "", "push_up", "", "clap"]

KEYPOINT_DICT = [
    [0, "nose", "코"], [1, "L_eye_inner", "왼 눈 안"], [2, "L_eye", "왼 눈"],
    [3, "L_eye_outer", "왼 눈 바깥"], [4, "R_eye_inner", "오른 눈 안"], [5, "R_eye", "오른 눈"],
    [6, "R_eye_outer", "오른 눈 바깥"], [7, "L_ear", "왼 귀"], [8, "R_ear", "오른 귀"],
    [9, "mouth_L", "입 왼쪽"], [10, "mouth_R", "입 오른쪽"], [11, "L_shoulder", "왼쪽 어깨"],
    [12, "R_shoulder", "오른쪽 어깨"], [13, "L_elbow", "왼쪽 팔꿈치"], [14, "R_elbow", "오른쪽 팔꿈치"],
    [15, "L_wrist", "왼쪽 손목"], [16, "R_wrist", "오른쪽 손목"], [17, "L_pinky", "왼손 소지"],
    [18, "R_pinky", "오른손 소지"], [19, "L_index", "왼손 검지"], [20, "R_index", "오른손 검지"],
    [21, "L_thumb", "왼손 엄지"], [22, "R_thumb", "오른손 엄지"], [23, "L_hip", "왼쪽 골반"],
    [24, "R_hip", "오른쪽 골반"], [25, "L_knee", "왼쪽 무릎"], [26, "R_knee", "오른쪽 무릎"],
    [27, "L_ankle", "왼쪽 발목"], [28, "R_ankle", "오른쪽 발목"], [29, "L_heel", "왼발 뒷꿈치"],
    [30, "R_heel", "오른발 뒷꿈치"], [31, "L_foot_index", "왼발 끝"], [32, "R_foot_index", "오른발 끝"]]

VECTOR_PAIR = [
    [0, 12, "Nose_RShoulder", "코→오른 어깨"],
    [12, 14, "RShoulder_RElbow", "오른 어깨→오른 팔꿈치"],
    [14, 16, "RElbow_RWrist", "오른 팔꿈치→오른 손목"],
    [0, 11, "Nose_LShoulder", "코→왼 어깨"],
    [11, 13, "LShoulder_LElbow", "왼 어깨→왼 팔꿈치"],
    [13, 15, "LElbow_LWrist", "왼 팔꿈치→왼 손목"],
    [12, 24, "RShoulder_RHip", "오른 어깨→오른 골반"],
    [24, 26, "RHip_RKnee", "오른 골반→오른 무릎"],
    [26, 28, "RKnee_RAnkle", "오른 무릎→오른 발목"],
    [11, 23, "LShoulder_LHip", "왼 어깨→왼 골반"],
    [23, 25, "LHip_LKnee", "왼 골반→왼 무릎"],
    [25, 27, "LKnee_LAnkle", "왼 무릎→왼 발목"]]
    
def plotConfusionMatrix(cfm, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(cfm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(cfm)):
        n = sum(cfm[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = cfm.max() / 2.
    if normalize:
        for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
            plt.text(j, i, '{:.1f}'.format(cfm[i, j] * 100 / n), horizontalalignment="center", color="white" if cfm[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
            plt.text(j, i, cfm[i, j], horizontalalignment="center", color="white" if cfm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def detectPose(frame, pose_setting):
    # 사진 BGR에서 RGB로 변환
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 성능 향상을 위해 이미지 쓰기 불가로 참조 전달
    frame.flags.writeable = False
    # 랜드마크 검출
    results = pose_setting.process(frame)
    # 다시 이미지 쓰기 가능으로 변경
    frame.flags.writeable = True
    # 사진 RGB에서 BGR로 변환
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # pose.process에서 랜드마크를 찾았다면
    try:
        landmarks = results.pose_landmarks.landmark
        # 이미지에 랜드마크 위치마다 표시
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Plot pose world landmarks
        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    # 랜드마크 검출 실패
    except AttributeError: landmarks = None
        
    return frame, landmarks

def calculateVectors(landmarks, vectors):
    for pair in VECTOR_PAIR:
        x_vector_normalized = (landmarks[pair[0]].x - landmarks[pair[1]].x)
        y_vector_normalized = (landmarks[pair[0]].y - landmarks[pair[1]].y)
        vectors.append(x_vector_normalized)
        vectors.append(y_vector_normalized)
    return vectors

def predictPose(model, vectors, all_frame_labels):
    x_input = np.array(vectors)
    x_input = x_input.reshape((1, 24, 1))
    # one-hot에 대해 레이블 분류 확률 array
    result_array = model.predict(x_input)
    # result_array에서 가장 높은 값이 레이블 index
    result_label = LABEL_LIST_string[np.argmax(result_array[0])]
    print(result_array[0], result_label, "\n")
    
    if result_array[0][0] >= 0.08:
        result_label = "stand"

    cv2.putText(frame, result_label, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0), 2, cv2.LINE_AA)
    
    return frame

## LSTM

# 합쳐진 CSV 파일 로드
merged_df = pd.read_csv(##DATA_CSV##)
# 의미없는 열 drop
merged_df = merged_df.drop(merged_df.columns[[0]], axis=1)
merged_df.describe()

# value/ target 따로 분리
df_X = merged_df.drop(["action"], axis=1)
df_Y = merged_df[["action"]]

# 현재 레이블링 되어 있는 액션
LABEL_LIST = []
LABEL_LIST_string = []
for i  in sorted(merged_df['action'].unique()): 
    print(i, ALL_LABEL_LIST[i])
    LABEL_LIST.append(i)
    LABEL_LIST_string.append(ALL_LABEL_LIST[i])

df_Y = df_Y.replace(6, 5)
df_Y = df_Y.replace(7, 6)
df_Y = df_Y.replace(8, 7)
df_Y = df_Y.replace(11, 8)
df_Y = df_Y.replace(13, 9)

# 파라미터 설정
n_input = 24  # 입력층 노드 갯수
n_labels = len(LABEL_LIST)
batch_size = 128
epochs = 100
learning_rate = 0.005

# train test 나누기
X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size = 0.2, random_state = 1)

# numpy형만 가져오기
X_train, X_test, Y_train, Y_test = X_train.values, X_test.values, Y_train.values, Y_test.values

print("Original :", df_X.shape, df_Y.shape)
print("Train :", X_train.shape, Y_train.shape)
print("Test :", X_test.shape, Y_test.shape)

X_train = X_train.reshape(X_train.shape[0], n_input, 1)
X_test = X_test.reshape(X_test.shape[0], n_input, 1)

# 종속변수 데이터 ONE-HOT 전처리
Y_train_onehot = np_utils.to_categorical(Y_train, n_labels)
Y_test_onehot = np_utils.to_categorical(Y_test, n_labels)
print(Y_train_onehot.shape, Y_test_onehot.shape)

## Hyper Parameters Tuning -> Keras Tuner

def create_model(hyper_params):
    K.clear_session()
    model = Sequential()
    n_hidden = hyper_params.Int('n_hidden', min_value=32, max_value=256, step=32)
    dropout_rate = hyper_params.Float('dropout_rate', min_value=0, max_value=0.3, step=0.1)
    learning_rate = hyper_params.Float('learning_rate', min_value=0, max_value=0.01, step=0.005)
    
    #dropout_rate2 = hyper_params.Float('dropout_rate2', min_value=0, max_value=0.3, step=0.1)
    #model.add(Conv1D(n_hidden, kernel_size=3, activation = 'tanh', input_shape=(24, 1)))
    #model.add(Dropout(dropout_rate_))
    model.add(CuDNNLSTM(n_hidden, input_shape=(n_input, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_labels, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                  ,metrics = ['accuracy'])
    model.summary()
    return model

# hyperband 튜너를 사용한 인스턴스화
tuner = keras_tuner.Hyperband(create_model,
                     objective = 'val_accuracy',
                     max_epochs = 2,
                     directory = 'saved_models',
                     project_name = 'keras_tuner_simplemodel')

# model.fit과 동일
tuner.search(x = X_train, y = Y_train_onehot,
             validation_data = (X_test, Y_test_onehot),
             epochs=epochs, use_multiprocessing=True, workers=6)

# 결과
tuner.results_summary()

# 최적 하이퍼 파라미터 가져오기
OptimalHyperParams = tuner.get_best_hyperparameters(num_trials=1)[0]
print("최적 은닉층 노드 수: {}".format(OptimalHyperParams.get("n_hidden")))
print("최적 드랍아웃률: {}".format(OptimalHyperParams.get("dropout_rate")))
print("최적 학습률 : {}".format(OptimalHyperParams.get("learning_rate")))

## Train

# 최적 모델 빌드 / 훈련
model = tuner.hypermodel.build(OptimalHyperParams)

# early stop / 최적 모델 상태 저장 설정
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
date = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')
save_path = f'saved_models\\SIMPLE_VECTOR_{date}-EPOCHS_{epochs}-BATCH_{batch_size}.hdf5'
checkpoint_callback = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# 모델 훈련
history = model.fit(x=X_train, y=Y_train_onehot,
              validation_data=(X_test, Y_test_onehot),
              batch_size=batch_size, epochs=epochs, 
              shuffle=True, verbose=1, callbacks=[early_stop, checkpoint_callback])

# 평가
# 최고 상태 model 로드
model = tf.keras.models.load_model(save_path)
tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True, show_layer_names=True)

## Test

# 모델을 평가합니다
loss, acc = model.evaluate(X_test, Y_test_onehot, verbose=1)
print('모델의 정확도: {:.5f}%, 손실값: {:.5f}'.format(100*acc, loss))

fig, loss_ax = plt.subplots(figsize=(10,8))
acc_ax = loss_ax.twinx()

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

loss_ax.plot(train_loss,'y',label='Loss (Train)')
loss_ax.plot(val_loss,'r',label='Loss (Test)')
acc_ax.plot(train_accuracy,'b',label='Accuracy (Train)')
acc_ax.plot(val_accuracy,'g',label='Accuracy (Test)')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# Predict the values from the validation dataset
predictions = model.predict(X_test)
# Convert predictions classes to one hot vectors
predictions_classes = np.argmax(predictions, axis = 1)
# Convert validation observations to one hot vectors
true_labels = np.argmax(Y_test_onehot, axis = 1)
# compute the confusion matrix
cfm = metrics.confusion_matrix(true_labels, predictions_classes)
# plot the confusion matrix
plotConfusionMatrix(cfm, range(n_labels), normalize=False)

# ================= Accuracy =================
acc = metrics.accuracy_score(true_labels, predictions_classes)
# ================= Precision =================
prec = metrics.precision_score(true_labels, predictions_classes, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='weighted')
# ================= Recall =================
recall = metrics.recall_score(true_labels, predictions_classes, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='weighted')
# ================= F1-Score =================
f1score = metrics.f1_score(true_labels, predictions_classes, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='weighted')

print("Accuracy : {}".format(acc))
print("Precision : {}".format(prec))
print("Recall : {}".format(recall))
print("F1-Score : {}".format(f1score))

## MAIN
cap = cv2.VideoCapture(input_path)
out = cv2.VideoWriter(input_path+'_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                      cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
pose_setting = mp_pose.Pose(smooth_landmarks = True,
                            min_detection_confidence=confidence,
                            min_tracking_confidence=confidence,
                            model_complexity=model_number)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if type(frame) == type(None): break
        
    # 랜드마크 검출
    frame, landmarks = detectPose(frame, pose_setting)
    # 랜드마크 검출 되었을 시에
    if landmarks:
        # 검출된 랜드마크 벡터 변환
        vectors = calculateVectors(landmarks, [])
        frame = predictPose(model, vectors, all_frame_labels)
    # FPS 출력
    cv2.putText(frame, "FPS: {}".format(round(1./(time.time() - start_time), 1)),
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 250), 2, cv2.LINE_AA)
    # 종료버튼 설정
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    # 출력
    cv2.imshow('Google Mediapipe BlazePose', frame)
    # 저장
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()