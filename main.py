"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.

Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""


import parse, Graph, time

def generate_dataset(file_name: str, output_folder: str) -> None:
    """ Although dataset have been generated and provide in data.zip, TAs might still want to generate dataset from scratch.
    If this is the case, please run this funcion. 

    In order to run this, you should provided the path to the original Manami dataset.
    We have downloaded for you, which is called original.json in ./data

    Running time depends on your computer. It will varies from 20 second to 18 minutes.
    """
    t = time.process_time()

    parse.parse_json(file_name, output_folder + "/full.json", True, 0, 40000)
    parse.parse_json(file_name, output_folder + "/small.json", True)

    G = Graph.Load_Graph_Fast().load_anime_graph_multiprocess(output_folder + "/small.json")
    G.serialize(output_folder + "/small_graph.json")

    G = Graph.Load_Graph_Fast().load_anime_graph_multiprocess(output_folder + "/full.json")
    G.serialize(output_folder + "/full_graph.json")
    
    elapsed_time = time.process_time() - t
    print(f"Dataset generation finished within {elapsed_time} sec")

if __name__ == "__main__":
    generate_dataset("data/original.json", "data")