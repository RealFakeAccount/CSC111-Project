"""CSC111 Winter 2021 Final Project

This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
"""
from __future__ import annotations
from typing import Union


class Anime:
    """Represents a node in the graph structure

    Instance Attributes:
        - title: The title of the anime
        - url: The url to the home page of the anime on http://myanimelist.net
        - thumbnail: The url to the thumbnail of the anime
        - score: ...
        - detail: ...
        - neighbours: ...
    """
    title: str
    url: str
    thumbnail: str
    score: float
    detail: str
    neighbours: list[Anime]

    # Private Instance Attributes:
    #   - _tags: A dictionary mapping each map of the anime to a weighting from 0 to 1 (inclusive)
    _tags: dict[str, float]

    def __init__(self, data: dict[str, Union[str, float, list[str]]]):
        self.title = data['title']
        self.url = data['url']
        self.thumbnail = data['thumbnail']
        self.score = data['score']
        self.detail = data['detail']
        self._tags = _initialize_tags(data['tags'])

        self.neighbours = []

    def calculate_similarity(self, anime: Anime) -> float:
        """Calculate the similarity between this anime and the given anime.

        The similarity of two anime is defined to be the dot product of their

        Version 1: use numpy's dot product; assumes not many tags, so minimal error.

        Version 2: use self-defined functions to find dot product. Slower, but more reliable
        """
        vector_1, vector_2 = self._tags.keys(), anime._tags.keys()

        similarity = 0

        for tag in vector_1:
            if tag in vector_2:
                similarity += self._tags[tag] * anime._tags[tag]

        return similarity

    def get_all_tags(self) -> set[str]:
        """Returns a list of every tag associated to this anime
        """
        return set(self._tags)

    def set_tag_weighting(self, tag: str, new_weighting: float) -> None:
        """Set the weighting of a tag in self.tags
        """
        if tag in self._tags:
            self._tags[tag] = new_weighting
            self._normalize_tags()
        else:
            raise ValueError

    def _normalize_tags(self) -> None:
        """Normalize all tags so that the real Euclidean norm of the total tag vector is 1.

        In other words, divide the weight of each tag by the sum of the squares of all weights.
        """
        # sum_of_squares = 0
        #
        # for tag in self._tags.keys():
        #     sum_of_squares += self._tags[tag] ** 2
        #
        # for tag in self._tags.keys():
        #     self._tags[tag] = self._tags[tag] / sum_of_squares
        normalize_dict(self._tags)

    def get_tags(self) -> dict[str, float]:
        """Return the tags attached to self.
        """
        return self._tags


def normalize_dict(values: dict[str, float]) -> None:
    """This function mutates the values dict such that the sum of the squares of all the values
    is NORMALIZATION_CONSTANT
    """
    sum_of_squares = 0

    for tag in values.keys():
        sum_of_squares += values[tag] ** 2

    for tag in values.keys():
        values[tag] = values[tag] / sum_of_squares * NORMALIZATION_CONSTANT


def _initialize_tags(tags: list[str]) -> dict[str, float]:
    """Initializes the tags in anime so that they all have the same weight."""
    anime_tags = {}

    for tag in tags:
        anime_tags[tag] = 1.0

    normalize_dict(anime_tags)

    return anime_tags
