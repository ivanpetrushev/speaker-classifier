import librosa
import librosa.display
import os
import json
from pprint import pprint
import matplotlib.pyplot as plt

# go with WAV instead of MP3 for speed
AUDIO_FILE = "daily-2020-04-03.wav"
CLASSIFICATION_FILE = "daily-2020-04-03.json"
EXTRACTED_FILE = "daily-2020-04-03-extracted.json"
SEGMENT_LENGTH = 3
MIN_NUMBER_OF_SEGMENTS_PER_TAG = 50


def save_mfcc():
    data = {
        'mapping': [],
        'labels': [],
        'mfcc': []
    }

    # load previously classified data file
    with open(CLASSIFICATION_FILE, 'r') as file:
        classified_data = json.load(file)

    data_by_tag = {}

    for item in classified_data:
        if 'note' in item['data']:
            tag = item['data']['note']
        else:
            tag = 'unknown'
        if tag not in data_by_tag:
            data_by_tag[tag] = {
                'tag': tag,
                'total_duration': 0,
                'segments_count': 0,
                'segments': []
            }

        offset = round(item['start'], 2)
        end = round(item['end'], 2)
        duration = round(end - offset, 2)
        while True:
            if end - offset < SEGMENT_LENGTH:
                break
            data_by_tag[tag]['segments_count'] += 1
            data_by_tag[tag]['total_duration'] += SEGMENT_LENGTH
            data_by_tag[tag]['segments'].append({
                'offset': offset,
                'duration': SEGMENT_LENGTH
            })
            offset += SEGMENT_LENGTH

    # remove tags with too few segments
    for tag in list(data_by_tag):
        if data_by_tag[tag]['segments_count'] < MIN_NUMBER_OF_SEGMENTS_PER_TAG:
            print('Deleting tag: {}, not enough segments (had {})'.format(tag, data_by_tag[tag]['segments_count']))
            del data_by_tag[tag]

    print('Tags survived culling:', list(data_by_tag))
    # pprint(data_by_tag, indent=4)

    for i, tag in enumerate(data_by_tag):
        data['mapping'].append(tag)
        print('Processing tag: {}'.format(tag))
        for j, segment in enumerate(data_by_tag[tag]['segments']):
            print('Tag {}, segment {}/{}'.format(i, j, len(data_by_tag[tag]['segments'])))
            y, sr = librosa.load(AUDIO_FILE, offset=segment['offset'], duration=SEGMENT_LENGTH)
            mfccs = librosa.feature.mfcc(y, n_mfcc=13, n_fft=512, hop_length=2048)
            data['mfcc'].append(mfccs.tolist())
            data['labels'].append(i)

    # save extracted useful data to json
    with open(EXTRACTED_FILE, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    save_mfcc()
