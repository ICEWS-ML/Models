# I tried a number of different preprocessing steps, a record of unused functions is kept here

from preprocess import get_data, preprocess_sampler, summary_statistics
import numpy as np

from collections import deque, Counter


# returns data aggregated by month. The measures are column names that have 'unique' lists in the summary_statistics
def get_month(measures):

    month_data = []
    date = None

    for observation in get_data():
        if date is None:
            date = observation['Event Date'].replace(day=1)

        if observation['Event Date'].replace(day=1) != date:
            output = {'Event Date': date}
            for measure in ['PLOVER', *measures]:

                counts = Counter(record[measure] for record in month_data)
                output[measure] = np.array([counts[key] for key in summary_statistics[measure]['uniques']]) / len(month_data)

            yield output

            date = observation['Event Date'].replace(day=1)
            month_data.clear()
        month_data.append(observation)


def day_sampler():
    """Assuming observations from dataset are (x, y), group by the day"""
    sampler = preprocess_sampler(x_format='OneHot', y_format='Ordinal')
    buffer = [next(sampler)]

    for observation in sampler:
        if buffer[0][0][0] != observation[0][0]:
            yield buffer
            buffer = [observation]
        buffer.append(observation)
    yield buffer


def day_offset_sampler(window=1):
    """Assuming observations from dataset are (x, y), compute the density of the y's over the next n days

    Args:
        window (int): how many days to use in the density
    Yield:
        [(x₁, y₁), (x₂, y₂), ...)]: where ∀xᵢ are part of the same day, and y is a density over the next {offset} days
    """
    buffer = deque()

    sampler = preprocess_sampler(x_format='OneHot', y_format='Ordinal')

    def add_day(day):
        # indexed: [newest day][first observation][first element of train pair][date index]
        if len(buffer) and buffer[-1][0][0][0] == day[0][0]:
            buffer[-1].append(day)
        else:
            buffer.append([day])

    while len(buffer) <= window + 1:
        add_day(next(sampler))

    for observation in sampler:
        add_day(observation)

        if len(buffer) > window + 1:
            current_day = buffer.popleft()
            temp = buffer.pop()  # the last day only has one element outside the window
            expected = np.mean([j[1] for i in buffer for j in i], axis=0)
            buffer.append(temp)
            yield [(record[0], expected) for record in current_day]


# this iterates through records aggregated by month
# for i, observation in enumerate(get_month(['Source Base Sector', 'Target Base Sector'])):
#     print(f'\nObservation: {i}')
#     print(observation)
