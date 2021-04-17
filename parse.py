"""CSC111 Winter 2021 Final Project

This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
"""
import json

import time
from graph import load_anime_graph_multiprocess


def parse_json(file_name: str, output_file: str, verbose: bool = True, start: int = 12500,
               end: int = 13500) -> None:
    """Parse the anime dataset from file_name and write the output to output_file"""
    new_data = {}
    skipped = 0
    with open(file_name) as original_file:
        data = json.load(original_file)
        for i in range(start, min(end, len(data['data']))):
            anime = data['data'][i]
            if len(anime['tags']) == 0:
                skipped += 1
                continue  # remove the anime with 0 tags
            description = ''  # get_anime_description(anime['sources'][0])

            new_data[anime['title']] = {
                'title': anime['title'],
                'url': anime['sources'][0],
                'thumbnail': anime['picture'],
                'tags': anime['tags'],
                'detail': description
            }
            if verbose:
                e = min(end, len(data['data']))
                print(f'{round((i - start) / (e - start) * 100, 2)}%')

        with open(output_file, 'w') as new_file:
            print(f'Writing to {output_file}.. {skipped} are skipped because of 0 tag')
            json.dump(new_data, new_file)


def generate_dataset(file_name: str, output_folder: str) -> None:
    """Generate the complete graph data using the complete dataset.

    The datasets are already provided in data.zip. This function is for generating
    the dataset from scratch.

    In order to run this, you should provided the path to the original Manami dataset.
    We have downloaded for you at ./data/original.json

    WARNING: This function is very computationally heavy. Running time can vary from 20 second to
    18 minutes depending on your computer. For a 2015 MacBook Air, it overheated and got the fan
    running at full speed for 8 minutes before we decided to stop the process :(
    """
    t = time.process_time()

    parse_json(file_name, output_folder + '/full.json', True, 0, 40000)
    parse_json(file_name, output_folder + '/small.json', True)

    small_graph = load_anime_graph_multiprocess(output_folder + '/small.json')
    small_graph.serialize(output_folder + '/small_graph.json')
    print('Finish writing to small_graph')

    full_graph = load_anime_graph_multiprocess(output_folder + '/full.json')
    full_graph.serialize(output_folder + '/full_graph.json')
    print('Finish writing to full_graph')

    elapsed_time = time.process_time() - t
    print(f'Dataset generation finished within {elapsed_time}s')


def update_graph(data_folder: str) -> None:
    """Update the graph using saved feedback

    WARNING: This function is very computationally heavy. Running time can vary from 20 second to
    18 minutes depending on your computer.
    """
    small_graph = load_anime_graph_multiprocess(
        data_folder + '/small.json', data_folder + '/feedback.json'
    )
    small_graph.serialize(data_folder + '/small_graph.json')

    full_graph = load_anime_graph_multiprocess(
        data_folder + '/full.json', data_folder + '/feedback.json'
    )
    full_graph.serialize(data_folder + '/full_graph.json')


if __name__ == '__main__':
    parse_json('./data/original.json', './data/full.json', True, 0, 40000)
    parse_json('./data/original.json', './data/small.json', True)

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E9999', 'E9998'],
        'allowed-io': ['parse_json', 'get_anime_description'],
        'max-nested-blocks': 4
    })

    import python_ta.contracts
    python_ta.contracts.check_all_contracts()
