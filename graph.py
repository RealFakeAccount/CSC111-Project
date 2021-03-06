"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.
Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""

from typing import Union, Optional
import json
import multiprocessing
import os
import requests
from bs4 import BeautifulSoup
import plotly
import networkx as nx
from anime import Anime, NEIGHBOUR_LIMIT

MAX_HISTORY_LIMIT = 10


class Graph:
    """A graph of anime to represent the popularity and similarity network
    """

    # Private Instance Attributes:
    #   - _anime: A collection of the anime contained in this graph
    #   - _feedback: A list of tuples containing users' feedback
    _anime: dict[str, Anime]
    _feedback: list[tuple[str, str, str]]

    def __init__(self) -> None:
        self._anime = dict()
        self._feedback = list()

    def add_anime(self, title: str, data: dict[str, Union[str, list[str]]]) -> None:
        """Add an anime into this graph
        """
        if title not in self._anime:
            self._anime[title] = Anime(data)

    def add_neighbour(self, anime_title: str, neighbour: str) -> None:
        """Append the given neighbour in the given anime's list of neighbours
        Unlike adding an edge in a typical graph, we only add the given neighbour in the list of
        the given anime's neighbours, without adding the given anime in the list of the given
        neighbour's neighbours. This is because neighbours are sorted by the similarity.
        Add neighbour_title even if neighbour_title is not in self._anime, because we assume that
        neighbour_title will eventually be added to the graph, based on our data structure.
        Preconditions:
            - anime_title in self._anime
        """
        self._anime[anime_title].neighbours.append(self._anime[neighbour])

    def get_all_anime(self) -> list[str]:
        """Return a list of all the anime in this graph"""
        return list(self._anime)

    def get_anime_description(self, anime_title: str) -> str:
        """Get the description/synopsis of the given anime"""
        if anime_title in self._anime:
            url = self._anime[anime_title].url
            page = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
            })
            soup = BeautifulSoup(page.content, 'html.parser')
            if soup is None:
                return 'No details found for this anime'

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
                return 'No details found for this anime'
        else:
            raise ValueError

    def get_anime_thumbnail_url(self, anime_title: str) -> str:
        """Get the thumbnail_url of the given anime
        """
        if anime_title in self._anime:
            thumbnail_url = self._anime[anime_title].thumbnail
            return thumbnail_url
        else:
            return 'Anime title not found'

    def get_similarity(self, anime1: str, anime2: str) -> float:
        """Return the similarity between anime1 and anime2.
        The similarity is explained in the project report
        """
        if anime1 in self._anime and anime2 in self._anime:
            return self._anime[anime1].calculate_similarity(self._anime[anime2])
        else:
            raise ValueError

    def _insert_neighbour(self, anime1: str, anime2: str) -> None:
        """Insert anime2 into anime1.neighbours based on the similarity of anime2 with anime1.
        One of the prototype methods that we explored to add neighbours (and consequently, edges) to
        our graph, which was eventually never used in favour of faster methods.
        Consequently, Anime.insert_neighbour is never used either
        """
        if anime1 in self._anime and anime2 in self._anime:
            self._anime[anime1].insert_neighbour(self._anime[anime2])
        else:
            raise ValueError

    def adjust_weighting_v1(self, anime_title: str, tag: str, reaction: str = 'upvote') -> None:
        """Change the weighting of tag in anime_title based on the user reaction.

        A first attempt at adjusting the weighting of anime based on user reactions. This method
        worked on changing the weighting of a single tag in an anime, and, once we found an
        alternative, we stopped using this method. We decided to keep this method in the final
        version of the program to display our thought process, for posterity.

        Preconditions:
            - reaction in {'upvote', 'downvote'} # decide on the name later
            - anime in self._anime
        """
        if anime_title in self._anime and reaction == 'upvote':
            self._anime[anime_title].adjust_tag_weighting(tag, 1.1)
        elif anime_title in self._anime and reaction == 'downvote':
            self._anime[anime_title].adjust_tag_weighting(tag, 0.9)
        else:
            raise ValueError

    def adjust_weighting(self, anime1: str, anime2: str, reaction: str = 'upvote') -> None:
        """Based on the user's reaction of how good anime2 is as a recommendation of anime1,
        adjust the tag weightings in anime1
        Preconditions:
            - reaction in {'upvote', 'downvote'}
        """
        if anime1 in self._anime and anime2 in self._anime:
            self._anime[anime1].adjust_from_feedback(self._anime[anime2], reaction)
        else:
            raise ValueError

    def sort_neighbours_multiprocess(self) -> None:
        """Calculate and sort the similarity between each anime pair and add the neighbours
        for each anime in descending order.
        This method is the same as self.calculate_neighbours, except that this method uses
        multiprocessing to speed up the process.
        """
        anime_list = list(self._anime.values())
        p = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        res = p.map(self._sort_neighbours, anime_list)
        p.close()
        p.join()

        self._anime = {anime_list[i].title: res[i] for i in range(0, len(anime_list))}

    def _sort_neighbours(self, anime: Anime) -> Anime:
        """Sort and reassigns the neighbours of the given anime and return the anime
        Warning: this method uses heavy computation and initializes the edges between anime.
        It is not meant to be accessible when the user's session is ongoing, and should only be
        used when the user quits their session.
        Preconditions:
            - anime in self._anime
        """
        anime.neighbours = sorted(list(self._anime.values()), key=anime.calculate_similarity,
                                  reverse=True)[:NEIGHBOUR_LIMIT]
        return anime

    def serialize(self, output_file: str) -> None:
        """Save the data of each Anime in this graph into an output file"""
        neighbours = {}

        for anime_name in self._anime:
            neighbours[anime_name] = {
                'title': self._anime[anime_name].title,
                'neighbours': [],
                'url': self._anime[anime_name].url,
                'thumbnail': self._anime[anime_name].thumbnail,
                'detail': self._anime[anime_name].detail,
                'tags': {tag: self._anime[anime_name].get_tag_weight(tag) for tag in
                         self._anime[anime_name].get_all_tags()}
            }
            for neighbour in self._anime[anime_name].neighbours:
                neighbours[anime_name]['neighbours'].append(neighbour.title)

        with open(output_file, 'w') as new_file:
            json.dump(neighbours, new_file)

    def prediction(self, past_choices: list[list[Anime]], options: list[Anime],
                   curr_anime: str) -> list[Anime]:
        """ Return a prediction of which anime in options the user is likely to choose, given all
        previous choices made and the current anime.
        Preconditions:
            - all(len(lst) == 2 for lst in past_choices)
        """
        if past_choices == []:
            return options
        else:
            prediction_weights = self._get_prediction_weights(curr_anime, past_choices)
            return sorted(options,
                          key=lambda anime: anime.predict_similarity(prediction_weights),
                          reverse=True)

    def store_history(self, curr_anime: str, rec_anime: str, store_file: str) -> None:
        """Store which anime the user searched for (curr_anime), and which anime they visited out
        of the recommendations (rec_anime) in the json file store_file.
        Preconditions:
            - curr_anime in self._anime and rec_anime in self._graph
            - store_file is a json file storing a single list argument
        """
        if curr_anime in self._anime and rec_anime in self._anime and os.path.exists(store_file):
            with open(store_file, 'a+') as json_file:
                data = json.load(json_file)
                if len(data) == 10:
                    data.pop(0)
                data.append([curr_anime, rec_anime])
                json.dump(data, json_file)
        elif curr_anime in self._anime and rec_anime in self._anime:
            with open(store_file, 'a+') as json_file:
                data = [curr_anime, rec_anime]
                json.dump([data], json_file)

    def _get_prediction_weights(self, curr_anime: str, past_choices: list[list[Anime]]) \
            -> dict[str, float]:
        """Get the weightings required to make the predictions
        For each list lst in past_choices, lst[0] denotes an anime that the user searched for, while
        lst[1] denotes the anime that the user clicked on next.
        Preconditions:
            - all(len(lst) == 2 for lst in past_choices)
        """
        prediction_weights = {initial_tag: 1 for initial_tag in
                              self._anime[curr_anime].get_all_tags()}
        for pair in past_choices:
            for tag in self._anime[curr_anime].get_all_tags():
                if tag in pair[0].get_all_tags() and tag in pair[1].get_all_tags():
                    prediction_weights[tag] += 1
        return prediction_weights

    def get_related_anime(self, anime_title: str, limit: int = 5,
                          visited: Optional[set[str]] = None) -> list[Anime]:
        """Return a list of up to <limit> anime that are related to the given anime,
        ordered by their similarity in descending order.
        The similarity is explained in the project report and in the Anime.py file.
        Raise ValueError if anime_title is not in the graph.
        """
        if visited is None:
            visited = set()

        if anime_title in self._anime:
            anime = self._anime[anime_title]

            res = []
            for i in range(NEIGHBOUR_LIMIT):
                if anime.neighbours[i].title not in visited and len(res) < limit:
                    res.append(self._anime[anime.neighbours[i].title])
            return res
        else:
            raise ValueError

    def add_connection(self, graph: nx.Graph(), cur_anime_title: str, det_anime_title: str) -> None:
        """Add an edge to a given networkx graph
        Preconditions:
            - cur_anime_tile in self._anime
            - det_anime_tile not in self._anime
        """
        if cur_anime_title in self._anime and det_anime_title in self._anime:
            if cur_anime_title != det_anime_title:
                graph.add_node(det_anime_title, kind=str)
                graph.add_edge(cur_anime_title, det_anime_title)
        else:
            raise ValueError

    def _get_all_edges_pos(self, graph: nx.Graph, nxg: dict) -> tuple[
        list[list], list[list], tuple[list[Optional[float]], list[Optional[float]]], list[
            Optional[str]]]:
        """Get all edges position in networkx graph and return a tuple of edges position in x-y
        dimension
        """
        x_edge_pos = []
        y_edge_pos = []
        x_mid_pos = []
        y_mid_pos = []
        edge_similarity = []
        for edge in graph.edges:
            x0, y0 = nxg[edge[0]][0], nxg[edge[0]][1]

            x1, y1 = nxg[edge[1]][0], nxg[edge[1]][1]

            x_edge_pos.extend([x0, x1, None])
            y_edge_pos.extend([y0, y1, None])

            x_mid_pos.extend([(x0 + x1) / 2, None])
            y_mid_pos.extend([(y0 + y1) / 2, None])

            edge_sim = self._anime[edge[0]].calculate_similarity(self._anime[edge[1]])

            edge_similarity.extend(
                [f'Similarity Score: {edge_sim} Between {edge[0]} and {edge[1]}', None])

        return (x_edge_pos, y_edge_pos, (x_mid_pos, y_mid_pos), edge_similarity)

    def draw_graph(self, anime_title: str, depth: int, limit: int) -> plotly.graph_objs.Figure():
        """Draw a plotly graph centered around the given anime title
        Preconditions:
            - depth <= 5 # This will be handled by the slider on the website
        """
        graph_layout = plotly.graph_objs.Layout(
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            height=1080
        )

        visited = set()
        visited.add(anime_title)
        graph = nx.Graph()
        shell = [[anime_title], []]  # [[center of graph], [other nodes]]
        queue = [(anime_title, 0)]  # title, depth
        while len(queue) != 0:
            cur = queue[0]
            queue.pop(0)

            for i in self.get_related_anime(cur[0], limit=limit, visited=visited):
                if i.title not in visited:
                    visited.add(i.title)

                    shell[1].append(i.title)
                    self.add_connection(graph, cur[0], i.title)
                    if cur[1] < depth - 1:
                        queue.append((i.title, cur[1] + 1))

        if shell[0] != shell[1]:
            nxg = nx.drawing.layout.spring_layout(graph)
        else:
            single_node = plotly.graph_objs.Scatter(
                x=[0],
                y=[0],
                hovertext=anime_title,
                mode="markers",
                name="nodes",
                marker={'size': 25, 'color': 'Red'}
            )
            figure = plotly.graph_objs.Figure(data=single_node, layout=graph_layout)
            return figure

        x_node_pos = [nxg[key][0] for key in graph.nodes if key != anime_title]
        y_node_pos = [nxg[key][1] for key in graph.nodes if key != anime_title]
        node_hover = [key for key in graph.nodes if key != anime_title]

        x_edge_pos, y_edge_pos, mid_pos, similarity = self._get_all_edges_pos(graph, nxg)

        all_traces = []

        central_node_trace = plotly.graph_objs.Scatter(
            x=[nxg[anime_title][0]],
            y=[nxg[anime_title][1]],
            hovertext=[anime_title],
            mode="markers",
            name="nodes",
            marker={'size': 25, 'color': 'Red'}
        )

        all_traces.append(central_node_trace)

        nodes_trace = plotly.graph_objs.Scatter(
            x=x_node_pos,
            y=y_node_pos,
            hovertext=node_hover,
            mode="markers",
            name="nodes",
            marker={'size': 25, 'color': 'LightSkyBlue'}
        )

        all_traces.append(nodes_trace)

        edges_trace = plotly.graph_objs.Scatter(
            x=x_edge_pos,
            y=y_edge_pos,
            mode="lines",
            name="edges",
            line=dict(color='rgb(210,210,210)', width=1),
            hoverinfo="none"
        )

        all_traces.append(edges_trace)

        hover_trace = plotly.graph_objs.Scatter(
            x=mid_pos[0],
            y=mid_pos[1],
            hovertext=similarity,
            name='similarity',
            mode='markers',
            hoverinfo="text",
            marker={'size': 5, 'color': 'Black'}
        )

        all_traces.append(hover_trace)

        figure = plotly.graph_objs.Figure(data=all_traces, layout=graph_layout)

        return figure

    def get_anime(self, anime_title: str) -> Anime:
        """Return the _anime attribute of the graph"""
        if anime_title in self._anime:
            return self._anime[anime_title]
        else:
            raise ValueError

    def store_feedback(self, reaction: str, curr_anime: str, feedback_anime: str) -> None:
        """Store the user's feedback to this graph
        Preconditions:
            - curr_anime in self._anime
            - feedback_anime in self._anime
            - reaction in {'upvote', 'downvote'}
        """
        self._feedback.append((curr_anime, feedback_anime, reaction))

    def dump_feedback_to_file(self, output_file: str) -> None:
        """Save the user feedback to an output file"""
        feedbacks = {}

        counter = 0
        for item in self._feedback:
            feedbacks[counter] = {
                'anime1': item[0],
                'anime2': item[1],
                'value': item[2]
            }
            counter += 1

        with open(output_file, 'w') as new_file:
            json.dump(feedbacks, new_file)

    def implement_feedback(self) -> None:
        """Mutate the anime in self._anime according to the feedback received.
        This method also mutates self._feedback so that, by the time this function has finished
        executing, self._feedback is empty.
        """
        if self._feedback != []:
            length = len(self._feedback)

            for _ in range(length):
                anime1, anime2, response = self._feedback.pop(0)
                self.adjust_weighting(anime1, anime2, response)

        assert self._feedback == []


def load_from_serialized_data(file_name: str) -> Graph:
    """Return the anime graph corresponding to the given serialized dataset
    Preconditions:
        - file_name is the path to a json file corresponding to the anime data format described in
      the project report
    """
    anime_graph = Graph()

    with open(file_name) as json_file:
        data = json.load(json_file)
        for title in data:
            anime_graph.add_anime(title, data[title])

        for title in data:
            for neighbour in data[title]['neighbours']:
                anime_graph.add_neighbour(title, neighbour)

    return anime_graph


def load_anime_graph_multiprocess(file_name: str, feedback: str = '') -> Graph:
    """Return the anime graph corresponding to the given dataset
    WARNING: This function is very computationally heavy. Running time can vary from 20 second to
    18 minutes depending on your computer. When running this with the full dataset on a 2015 MacBook
    Air, it overheated and got the fan running at full speed for 8 minutes before we decided to stop
    the process :(
    For another context: it takes about 17s using 3900x.
    Preconditions:
        - file_name is the path to a json file corresponding to the anime data
        format described in the project report
    """
    anime_graph = Graph()

    with open(file_name) as json_file:
        data = json.load(json_file)
        for title in data:
            anime_graph.add_anime(title, data[title])

    if feedback != '' and os.path.exists(feedback):
        with open(feedback) as json_file:
            data = json.load(json_file)
            for item in data:
                anime_graph.store_feedback(
                    data[item]['value'],
                    data[item]['anime1'],
                    data[item]['anime2']
                )

    anime_graph.implement_feedback()
    anime_graph.sort_neighbours_multiprocess()
    return anime_graph


if __name__ == "__main__":

    # This code was used to check whether our program worked and whether it worked in a reasonable
    # period of time.
    # import time
    # t = time.process_time()
    # graph = load_from_serialized_data("data/full_graph.json")
    # print(sum(len(i._tags) == 0 for i in graph._anime.values()))
    # elapsed_time = time.process_time() - t
    # print(f"process takes {elapsed_time} sec")

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E9999', 'E9998', 'E1136'],
        'max-nested-blocks': 4
    })

    import python_ta.contracts
    python_ta.contracts.check_all_contracts()
