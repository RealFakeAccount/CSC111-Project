"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.

Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""


import parse
from graph import LoadGraphFast
import time


def generate_dataset(file_name: str, output_folder: str) -> None:
    """ The datasets are already provided in data.zip, but run this if you want to generate
    the dataset from scratch.

    In order to run this, you should provided the path to the original Manami dataset.
    We have downloaded for you at ./data/original.json

    Running time depends on your computer. It varies from 20 second to 18 minutes.
    """
    t = time.process_time()

    parse.parse_json(file_name, output_folder + '/full.json', True, 0, 40000)
    parse.parse_json(file_name, output_folder + '/small.json', True)

    graph = LoadGraphFast().load_anime_graph_multiprocess(output_folder + "/small.json")
    graph.serialize(output_folder + "/small_graph.json")
    print("Finish writing to small_graph")

    graph = LoadGraphFast().load_anime_graph_multiprocess(output_folder + "/full.json")
    graph.serialize(output_folder + "/full_graph.json")
    print("Finish writing to full_graph")

    elapsed_time = time.process_time() - t
    print(f'Dataset generation finished within {elapsed_time}s')


if __name__ == '__main__':
    generate_dataset('data/original.json', 'data')
