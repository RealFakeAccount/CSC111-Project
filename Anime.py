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

    Private Instance Attributes:
        - _tags: A dictionary mapping each map of the anime to a weighting from 0 to 1 (inclusive)
    """
    title: str
    url: str
    thumbnail: str
    score: float
    detail: str
    _tags: dict[str, float]

    def __init__(self, data: dict[str, Union[str, float, list[str]]]):
        self.title = data['title']
        self.url = data['url']
        self.thumbnail = data['thumbnail']
        self.score = data['score']
        self.detail = data['detail']
        # self._tags = data['tags']

    def calculate_similarity(self, anime: Anime) -> float:
        """Calculate the similarity between this anime and the given anime
        """
        ...

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
        """."""
        ...
