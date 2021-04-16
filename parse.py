"""CSC111 Winter 2021 Final Project

This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
"""
import json


def parse_json(file_name: str, output_file: str, verbose: bool = True, start: int = 12500,
               end: int = 13500) -> None:
    """Parse the anime dataset from file_name and write the output to output_file"""
    new_data = {}
    cnt_skip = 0
    with open(file_name) as original_file:
        data = json.load(original_file)
        for i in range(start, min(end, len(data['data']))):
            anime = data['data'][i]
            if len(anime['tags']) == 0:
                cnt_skip += 1
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
            print(f'Writing to {output_file}.. {cnt_skip} are skipped because of 0 tag')
            json.dump(new_data, new_file)


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
