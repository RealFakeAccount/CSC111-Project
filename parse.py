"""CSC111 Winter 2021 Final Project

This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
"""
import json

from networkx.classes import graph
from graph import LoadGraphFast, load_from_serialized_data
import time


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
    """ The datasets are already provided in data.zip, but run this if you want to generate
    the dataset from scratch.

    In order to run this, you should provided the path to the original Manami dataset.
    We have downloaded for you at ./data/original.json

    WARNING: Running time depends on your computer. It varies from 20 second to 18 minutes.
    """
    t = time.process_time()

    parse_json(file_name, output_folder + '/full.json', True, 0, 40000)
    parse_json(file_name, output_folder + '/small.json', True)

    graph = LoadGraphFast().load_anime_graph_multiprocess(output_folder + '/small.json')
    graph.serialize(output_folder + '/small_graph.json')
    print('Finish writing to small_graph')

    graph = LoadGraphFast().load_anime_graph_multiprocess(output_folder + '/full.json')
    graph.serialize(output_folder + '/full_graph.json')
    print('Finish writing to full_graph')

    elapsed_time = time.process_time() - t
    print(f'Dataset generation finished within {elapsed_time}s')

def update_graph(output_folder: str) -> None:
    """
    update the graph using saved feedback 
    
    WARNING: Running time depends on your computer. It varies from 20 second to 18 minutes.
    """
    G = LoadGraphFast().load_anime_graph_multiprocess(
        output_folder + '/small.json', output_folder + '/feedback.json'
    )
    G.serialize(output_folder + '/small_graph.json')

    G = LoadGraphFast().load_anime_graph_multiprocess(
        output_folder + '/full.json', output_folder + '/feedback.json'
    )
    G.serialize(output_folder + '/full_graph.json')
    

if __name__ == '__main__':
    parse_json('./data/original.json', './data/full.json', True, 0, 40000)
    parse_json('./data/original.json', './data/small.json', True)

    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 100,
    #     'disable': ['E9999'],
    #     'allowed-io': ['parse_json', 'get_anime_description'],
    #     'max-nested-blocks': 4
    # })
