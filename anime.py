"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials,
please do not consult the Course Syllabus.
Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""
from __future__ import annotations
from typing import Union
from math import sqrt

NORMALIZATION_SCALE = 1.0
NORMALIZATION_CONSTANT = sqrt(NORMALIZATION_SCALE)
NEIGHBOUR_LIMIT = 20


class Anime:
    """Represents an Anime as a vertex in the Anime graph
    Instance Attributes:
        - title: The title of the anime
        - url: The url to the home page of the anime on an anime website. Ex. http://myanimelist.net
        - thumbnail: The url to the thumbnail of the anime
        - detail: The synopsis of the anime
        - neighbours: A list of anime that are similar to this anime
    """

    title: str
    url: str
    thumbnail: str
    detail: str
    neighbours: list[Anime]

    # Private Instance Attributes:
    #   - _tags: A dictionary mapping each tag of the anime to a weighting from 0 to 1 (inclusive)
    _tags: dict[str, float]

    def __init__(self, data: dict[str, Union[str, list[str]]]) -> None:
        self.title = data['title']
        self.url = data['url']
        self.thumbnail = data['thumbnail']
        self.detail = data['detail']
        self._tags = _initialize_tags(data['tags'])
        self.neighbours = []

    def calculate_similarity(self, anime: Anime) -> float:
        """Calculate the similarity between this anime and the given anime.

        The similarity of two anime is defined to be the sum of the product of the weightings of the
        tags that they have in common. This is analogous to the dot product of two vectors, and,
        indeed, if we were using lists to represent the tags' weightings like mentioned in the
        project proposal, we could use numpy's in-built dot product.
        """
        vector_1, vector_2 = self._tags.keys(), anime._tags.keys()

        similarity = 0

        for tag in vector_1:
            if tag in vector_2:
                similarity += self._tags[tag] * anime._tags[tag]

        return similarity

    def get_all_tags(self) -> set[str]:
        """Return a set of all the tags associated with this anime"""
        return set(self._tags)

    def get_tag_weight(self, tag: str) -> float:
        """Return the weighting of the given tag in this anime"""
        if tag in self._tags:
            return self._tags[tag]
        else:
            raise ValueError

    def adjust_tag_weighting(self, tag: str, scale: float) -> None:
        """Adjust the weighting of a tag in this anime by the given scale
        Scales larger than 1 increase the weighting; scales less than 1 decrease the weighting.
        A scale of 1 does not change the weighting
        """
        if tag in self._tags:
            self._tags[tag] *= scale
            self._normalize_tags()
        else:
            raise ValueError

    def _normalize_tags(self) -> None:
        """Normalize all tags so that the real Euclidean norm of the total tag vector is 1.
        In other words, divide the weight of each tag by the sum of the squares of all weights.
        """
        _normalize_dict(self._tags)

    def insert_neighbour(self, anime: Anime) -> None:
        """Insert anime into self.neighbours according to similarity."""
        # VERSION 1.
        self.neighbours.append(anime)
        self.neighbours.sort(key=self.calculate_similarity, reverse=True)
        self.neighbours = self.neighbours[:NEIGHBOUR_LIMIT]

        # VERSION 2.
        # if self.neighbours == []:
        #     self.neighbours.append(anime)
        #     return None

        # n = len(self.neighbours)

        # while self.calculate_similarity(anime) > self.calculate_similarity(anime.neighbours[n])
        # and n > 0:
        #     n -= 1

        # self.neighbours.insert(n, anime)

    def adjust_from_feedback(self, anime: Anime, feedback: str) -> None:
        """Readjust the weightings upon receiving a feedback for similarity with anime.
        Preconditions:
            - feedback in {'upvote', 'downvote'}
        """
        for tag in self._tags:
            if tag in anime._tags:
                if feedback == 'upvote':
                    self._tags[tag] = self._tags[tag] * 1.1
                else:
                    self._tags[tag] = self._tags[tag] * 0.9

        _normalize_dict(self._tags)

    def predict_similarity(self, prediction_weights: dict[str, float]) -> float:
        """Return how likely the user is to pick anime. This is NOT a measure of probability, and
        is hence not normalized.
        """
        sum_so_far = 0
        for tag in self.get_all_tags():
            if tag in prediction_weights:
                sum_so_far += prediction_weights[tag]
        return sum_so_far


def _normalize_dict(values: dict[str, float]) -> None:
    """Mutate the values of dict such that the sum of the squares of all the values
    is NORMALIZATION_CONSTANT.
    """
    sum_of_squares = 0

    for tag in values.keys():
        sum_of_squares += values[tag] ** 2

    for tag in values.keys():
        values[tag] = values[tag] / sum_of_squares * NORMALIZATION_CONSTANT


def _initialize_tags(tags: list[str]) -> dict[str, float]:
    """
    Given a list of tags, return a dict with <tag>:<value> pairs such that the values corresponding
    to each tag are the same and the sum of the squares of values is  NORMALIZATION_SCALE.
    This is used to initialize the tags in anime so that they all have the same weight.
    """
    anime_tags = {}

    for tag in tags:
        anime_tags[tag] = 1.0

    _normalize_dict(anime_tags)

    return anime_tags


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E1136', 'E9999'],
        'max-nested-blocks': 4
    })

    import python_ta.contracts
    python_ta.contracts.check_all_contracts()

    # import doctest
    # doctest.testmod()
