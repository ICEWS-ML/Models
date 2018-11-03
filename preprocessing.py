
import json
import os
from datetime import datetime

data_folder = 'data'
metadata_folder = 'metadata'


def coerce(key, value):
    datatypes = {
        'Event ID': str,
        'Event Date': datetime,
        'Source Name': str,
        'Source Sectors': str,
        'Source Country': str,
        'Event Text': str,
        'CAMEO Code': str,
        'Intensity': float,
        'Target Name': str,
        'Target Sectors': str,
        'Target Country': str,
        'Story ID': str,
        'Sentence Number': int,
        'Publisher': str,
        'City': str,
        'District': str,
        'Province': str,
        'Country': str,
        'Latitude': float,
        'Longitude': float
    }

    if datatypes[key] is datetime:
        return datetime.strptime(value, '%Y-%m-%d')
    try:
        return datatypes[key](value)
    except ValueError:
        if not value:
            return None
        return value


def get_data():
    # linearize the CAMEO lookup
    reformatter = {}
    with open(os.path.join(os.getcwd(), metadata_folder, 'action.json'), 'r') as actionfile:
        for alignment in json.load(actionfile):
            if 'PLOVER' in alignment and 'CAMEO' in alignment:
                reformatter[alignment['CAMEO']] = alignment['PLOVER']

    for dataset in os.listdir(os.path.join(os.getcwd(), data_folder)):
        with open(os.path.join(os.getcwd(), data_folder, dataset), 'r', encoding='utf8') as datafile:
            headers = next(datafile)[:-1].split('\t')

            for line in datafile:
                parsed = {head: coerce(head, value) for head, value in zip(headers, line[:-1].split('\t'))}

                if parsed['CAMEO Code'] not in reformatter:
                    continue

                parsed['PLOVER'] = reformatter[parsed['CAMEO Code']]
                yield parsed


for observation in get_data():
    print(observation)
