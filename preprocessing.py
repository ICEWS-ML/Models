
import json
import os
from datetime import datetime
from collections import Counter

data_folder = 'data'
metadata_folder = 'metadata'


# parses strings into date objects, strings, floats and ints
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
    # linearize the class lookups
    action_reformatter = {}
    with open(os.path.join(os.getcwd(), metadata_folder, 'action.json'), 'r') as actionfile:
        for alignment in json.load(actionfile):
            if 'PLOVER' in alignment and 'CAMEO' in alignment:
                action_reformatter[alignment['CAMEO']] = alignment['PLOVER']

    sector_reformatter = json.load(open(os.path.join(os.getcwd(), metadata_folder, 'sectors.json'), 'r'))

    def most_common_base_sector(value):
        if not value:
            return
        bases = [sector_reformatter[i] for i in parsed['Source Sectors'].split(',') if i in sector_reformatter]
        if bases:
            return Counter(bases).most_common(1)[0][0]

    for dataset in os.listdir(os.path.join(os.getcwd(), data_folder)):
        with open(os.path.join(os.getcwd(), data_folder, dataset), 'r', encoding='utf8') as datafile:

            headers = next(datafile)[:-1].split('\t')

            for line in datafile:
                parsed = {head: coerce(head, value) for head, value in zip(headers, line[:-1].split('\t'))}

                if parsed['CAMEO Code'] not in action_reformatter:
                    continue

                # construct a new column with the aggregated PLOVER code
                parsed['PLOVER'] = action_reformatter[parsed['CAMEO Code']]

                parsed['Source Base Sector'] = most_common_base_sector(parsed['Source Sectors'])
                parsed['Target Base Sector'] = most_common_base_sector(parsed['Target Sectors'])

                yield parsed


for observation in get_data():
    pass
