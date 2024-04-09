from tensorflow.keras.models import Sequential, model_from_json
import librosa
import keras
from keras.models import Sequential

import numpy as np

# loaded_model = model_from_json(loaded_model_json)
loaded_model = keras.models.load_model("./model.keras")

# load weights into new model
loaded_model.load_weights("./CNN_model_weights.h5")
print("Loaded model from disk")

import pickle

with open('./scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

print("Done")

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result


def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    
    return final_result


emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(path1):
    res=get_predict_feat(path1)
    predictions=loaded_model.predict(res)
    predictions = np.array(predictions)
    top_three_indices = predictions.argsort()[0][-1:][::-1]
    classes = []
    for i in top_three_indices:
        classes.append(emotions1[i+1])  
    print("predictions",predictions)
    return classes
    # y_pred = encoder2.inverse_transform(predictions)
    # print(y_pred[0][0])    