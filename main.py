"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.

Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""
from parse import generate_dataset, update_graph
from visualization import run_test_server

if __name__ == '__main__':
    # To test generating dataset, uncomment this line
    # generate_dataset('./data/original.json', 'data')
    run_test_server()
    # To update the graph using the feedbacks, uncomment this line
    update_graph("data/")