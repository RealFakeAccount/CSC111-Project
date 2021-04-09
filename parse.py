"""CSC111 Winter 2021 Final Project

This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
"""
import json


def parse_json(file_name: str, output_file: str) -> None:
    """Parse the anime dataset from file_name and write the output to output_file"""
    new_data = {}

    with open(file_name) as original_file:
        data = json.load(original_file)
        for anime in data['data']:
            new_data[anime['title']] = {
                'title': anime['title'],
                'url': anime['sources'][0],
                'thumbnail': anime['picture'],
                'tags': anime['tags']
            }

        with open(output_file, 'w') as new_file:
            json.dump(new_data, new_file)
