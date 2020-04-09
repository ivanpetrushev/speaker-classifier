import tensorflow.keras as keras
import sys
import librosa
# from pygame import mixer
import time
import numpy as np
import json
from subprocess import run

SEGMENT_LENGTH = 3
EXTRACTED_FILE = "data/extracted.json"

model_filename = sys.argv[1]
audio_filename = sys.argv[2]
audio_start = float(sys.argv[3])
audio_end = audio_start + 3

# load label mapping
with open(EXTRACTED_FILE, 'r') as file:
    data = json.load(file)
mapping = data['mapping']
print('Mapping', mapping)

# play audio portion for verification
# Sorry, I get tons of issues trying to make python to speak proper audio in whichever lib...
# mixer.init(frequency=48000)
# mixer.music.load(audio_filename)
# mixer.music.play(start=audio_start)
# time.sleep(3)
# mixer.music.stop()

# play audio segment via sox/play
cmd = ['play', audio_filename, 'trim', str(audio_start), '=' + str(audio_end)]
print("Play command: ", cmd)
run(cmd)

# calculate MFCCs
y, sr = librosa.load(audio_filename, offset=audio_start, duration=SEGMENT_LENGTH)
mfccs = librosa.feature.mfcc(y, n_mfcc=13, n_fft=512, hop_length=2048)

# load model
model = keras.models.load_model(model_filename)
mfccs = mfccs[..., np.newaxis]
mfccs = mfccs[np.newaxis, ...]
prediction = model.predict(mfccs)
prediction = prediction[0]
selected_idx = np.argmax(prediction)
print('prediction:', np.round(prediction, 2))
print('selected idx:', selected_idx)
print('mapping:', mapping[selected_idx])
