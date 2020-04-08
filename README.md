# Combine mp3 files
> recordings/mp3] $ sox  2020-04-04/*mp3 /output/path/daily-2020-04-04.mp3


# Prediction
> python3.6 predict.py [model filename] [audio filename] [audio start position]
> python3.6 predict.py model-2020-04-08_16\:31\:10.h5 daily-2020-04-04.wav 321
