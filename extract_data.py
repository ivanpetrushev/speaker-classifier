import librosa
import librosa.display
import os
import json
from pprint import pprint, pformat
import matplotlib.pyplot as plt

# go with WAV instead of MP3 for speed
CLASSIFICATION_SETS = ["daily-2020-04-03", "daily-2020-04-04", "daily-2020-04-05", "daily-2020-04-06"]
EXTRACTED_FILE = "extracted.json"
SEGMENT_LENGTH = 3
MIN_NUMBER_OF_SEGMENTS_PER_TAG = 100


def save_mfcc():
    data = {
        'mapping': [],
        'labels': [],
        'mfcc': []
    }
    classified_data = []

    # load previously classified data file
    for classification_set in CLASSIFICATION_SETS:
        filename = 'data/' + classification_set + '.json'
        print("Loading set: ", filename)
        with open(filename, 'r') as file:
            current_classified_data = json.load(file)
        for i, item in enumerate(current_classified_data):
            current_classified_data[i]['set'] = classification_set
        classified_data = classified_data + current_classified_data

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
                'duration': SEGMENT_LENGTH,
                'set': item['set']
            })
            offset += SEGMENT_LENGTH

    # display number of segments per tag
    counts_by_tag = []
    for tag in list(data_by_tag):
        counts_by_tag.append({
            'tag': tag,
            'segments_count': data_by_tag[tag]['segments_count']
        })
    counts_by_tag = sorted(counts_by_tag, key=lambda el: el['segments_count'], reverse=True)
    print('Number of segments')
    pprint(counts_by_tag, indent=4)

    # remove tags with too few segments
    print('Min number of segments per tag:', MIN_NUMBER_OF_SEGMENTS_PER_TAG)
    for tag in list(data_by_tag):
        if data_by_tag[tag]['segments_count'] < MIN_NUMBER_OF_SEGMENTS_PER_TAG:
            print('Deleting tag: {}, not enough segments (had {})'.format(tag, data_by_tag[tag]['segments_count']))
            del data_by_tag[tag]

    print('Tags survived culling:', list(data_by_tag))

    for i, tag in enumerate(data_by_tag):
        data['mapping'].append(tag)
        print('Processing tag: {}'.format(tag))
        for j, segment in enumerate(data_by_tag[tag]['segments']):
            print('Tag {}, segment {}/{}'.format(i, j, len(data_by_tag[tag]['segments'])), end='\r')
            audio_filename = 'data/' + segment['set'] + '.wav'
            y, sr = librosa.load(audio_filename, offset=segment['offset'], duration=SEGMENT_LENGTH)
            mfccs = librosa.feature.mfcc(y, n_mfcc=13, n_fft=512, hop_length=2048)
            data['mfcc'].append(mfccs.tolist())
            data['labels'].append(i)

    print('Writing data to:', EXTRACTED_FILE)
    # save extracted useful data to json
    with open(EXTRACTED_FILE, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    save_mfcc()
