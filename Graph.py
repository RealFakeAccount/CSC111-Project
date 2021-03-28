"""CSC111 Winter 2021 Final Project

This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
"""
from Anime import Anime
import plotly
from typing import Union
import json


class Graph:
    """A graph of anime to represent the popularity and similarity network

    Private Instance Attributes:
        - _anime: A collection of the anime contained in this graph.
    """
    _anime: dict[str, Anime]

    def __init__(self) -> None:
        self._anime = {}

    def add_anime(self, title: str, data: dict[str, Union[str, float, dict[str, float]]]) -> None:
        """Add an anime into this graph
        """
        if title not in self._anime:
            self._anime[title] = Anime(data)

    def get_similarity(self, anime1: str, anime2: str) -> float:
        """Return the similarity between anime1 and anime2.
        The similarity is explained in the project report
        """
        if anime1 in self._anime and anime2 in self._anime:
            return self._anime[anime1].calculate_similarity(self._anime[anime2])
        else:
            raise ValueError

    def get_related_anime(self, anime_title: str, limit: int = 20) -> list[Anime]:
        """Return a list of up to <limit> anime that are related to the given anime,
        ordered by their similarity in descending order.

        The similarity is explained in the project report
        """
        ...

    def draw_graph(self, anime_title: str, depth: int) -> plotly.graph_objs.Figure():
        """Draw a plotly graph centered around the given anime title

        Preconditions:
            - depth <= 5 # This will be handled by the slider on the website
        """
        ...

    def adjust_weighting(self, anime: Anime, tag: str, reaction: str = 'upvote') -> None:
        """
        Preconditions:
            - reaction in {'upvote', 'downvote'} # decide on the name later
        """
        if reaction == 'upvote':
            anime.set_tag_weighting(tag, ...)
        else:
            anime.set_tag_weighting(tag, ...)


def load_anime_graph(file_name: str) -> Graph:
    """Return the anime graph corresponding to the given dataset

    Preconditions:
        - file_name is the path to a json file corresponding to the anime data
          format described in the project report
    """
    anime_graph = Graph()

    with open(file_name) as json_file:
        data = json.load(json_file)
        for title in data:
            anime_graph.add_anime(title, data[title])

    return anime_graph
