# Workflow
1. Combine all directory recordings to single file:

> recordings/mp3] $ sox  2020-04-04/*mp3 /output/path/daily-2020-04-04.mp3

2. Move single file to js manual classifier:

> mv daily-2020-04-04.mp3 /path/to/js-audio-classifier

3. Classify audio:

> cd /path/to/js-audio-classifier
> vim app.js # set filename
> ./start-webserver.sh

4. Save exported JSON to /output/path/daily-2020-04-04.json

5. Save mp3 there and convert to WAV for speed:

> ffmpeg -i daily-2020-04-04.mp3 daily-2020-04-04.wav

6. Combine all exports to a single set:

Add new set to CLASSIFICATION_SETS in extract_data.json

> cd /output/path; python3.6 extract_data.py

7. Train NN on that single set:

> python3.6 build_nn.py


# Prediction
> python3.6 predict.py [model filename] [audio filename] [audio start position]
> python3.6 predict.py model-2020-04-08_16\:31\:10.h5 daily-2020-04-04.wav 321
