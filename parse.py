"""CSC111 Winter 2021 Final Project

This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
"""
import json
import requests
from bs4 import BeautifulSoup


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
                continue # remove the anime with 0 tags
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
            print(f"Writing to {output_file}.. {cnt_skip} are skipped because of 0 tag")
            json.dump(new_data, new_file)


def get_anime_description(url: str) -> str:
    """Scrape the description/synopsis of an anime from the given url
    """
    page = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    })
    soup = BeautifulSoup(page.content, 'html.parser')
    if url.startswith('https://anidb.net'):
        detail_raw = soup.find(itemprop='description')
    elif url.startswith('https://myanimelist.net'):
        detail_raw = soup.find(itemprop='description')
    elif url.startswith(' https://kitsu.io'):
        detail_raw = soup.find(id='ember66')
    elif url.startswith('https://anime-planet.com'):
        detail_raw = soup.find('div', class_='md-3-5').find('p')
    else:
        detail_raw = None

    if detail_raw is not None:
        return detail_raw.text
    else:
        print(f'Failed: {url}')
        return 'No details found for this anime'


if __name__ == '__main__':
    parse_json('data/original.json', 'full.json', True, 0, 40000)
    parse_json('data/original.json', 'small.json', True)

    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 100,
    #     'disable': ['E9999'],
    #     'allowed-io': ['parse_json', 'get_anime_description'],
    #     'max-nested-blocks': 4
    # })
