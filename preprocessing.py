
import json
import os
from datetime import datetime
from collections import Counter

data_folder = 'data'
metadata_folder = 'metadata'

statistics_path = 'statistics.json'


def data_filter(record):
    return record['Country'] == 'Nepal'


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

                if not data_filter(parsed):
                    continue

                if parsed['CAMEO Code'] not in action_reformatter:
                    continue

                # construct a new column with the aggregated PLOVER code
                parsed['PLOVER'] = action_reformatter[parsed['CAMEO Code']]

                parsed['Source Base Sector'] = most_common_base_sector(parsed['Source Sectors'])
                parsed['Target Base Sector'] = most_common_base_sector(parsed['Target Sectors'])

                yield parsed


def compute_statistics():
    print('Computing summary statistics about predictors')

    statistics = {
        'Event Date': {
            'count': 0,
            'mean': 0,
            'sq_diff': 0
        },
        'Latitude': {
            'count': 0,
            'mean': 0,
            'sq_diff': 0
        },
        'Longitude': {
            'count': 0,
            'mean': 0,
            'sq_diff': 0
        },
        'Source Base Sector': {
            'uniques': set()
        },
        'Target Base Sector': {
            'uniques': set()
        }
    }

    def update_continuous(value, params):
        if not value:
            return
        # Welford's Online Algorithm for computing mean/variance
        params['count'] += 1
        delta = value - params['mean']
        params['mean'] += delta / params['count']
        delta2 = value - params['mean']
        params['sq_diff'] += delta * delta2

    def update_categorical(value, params):
        if not value:
            return
        params['uniques'].add(value)

    for record in get_data():
        update_continuous(record['Event Date'].timestamp(), statistics['Event Date'])
        update_continuous(record['Latitude'], statistics['Latitude'])
        update_continuous(record['Longitude'], statistics['Longitude'])
        update_categorical(record['Source Base Sector'], statistics['Source Base Sector'])
        update_categorical(record['Target Base Sector'], statistics['Target Base Sector'])

    for variable in statistics.values():
        if 'sq_diff' in variable:
            variable['sample_variance'] = variable['sq_diff'] / (variable['count'] - 1)
        if 'uniques' in variable:
            variable['uniques'] = list(variable['uniques'])

    json.dump(statistics, open(os.path.join(os.getcwd(), statistics_path), 'w'), indent=4)


if not os.path.exists(os.path.join(os.getcwd(), statistics_path)):
    compute_statistics()

for observation in get_data():
    print(observation)
