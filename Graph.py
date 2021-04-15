"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.

Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""

from typing import Union
import json
import multiprocessing
import time
import plotly
import networkx as nx
from Anime import Anime, NEIGHBOUR_LIMIT
import parse


class Graph:
    """A graph of anime to represent the popularity and similarity network
    Private Instance Attributes:
        - _anime: A collection of the anime contained in this graph.
    """
    _anime: dict[str, Anime]

    def __init__(self) -> None:
        self._anime = dict()

    def add_anime(self, title: str, data: dict[str, Union[str, list[str]]]) -> None:
        """Add an anime into this graph
        """
        if title not in self._anime:
            self._anime[title] = Anime(data)

    def get_all_anime(self) -> list[str]:
        """Return a list of all the anime in this graph
        """
        return list(self._anime)

    def get_anime_description(self, title: str) -> str:
        """ return the name of one anime.
        return an empty string if not found
        """
        return parse.get_anime_description(
            self._anime[title].url) if title in self._anime else "Anime title not found"

    def get_similarity(self, anime1: str, anime2: str) -> float:
        """Return the similarity between anime1 and anime2.
        The similarity is explained in the project report
        """
        if anime1 in self._anime and anime2 in self._anime:
            return self._anime[anime1].calculate_similarity(self._anime[anime2])
        else:
            raise ValueError

    def _insert_neighbour(self, anime1: Anime, anime2: Anime) -> None:
        """bleh"""
        anime1.insert_neighbour(anime2)

    def _adjust_positive_feedback(self, anime1: Anime, anime2: Anime) -> None:
        """Readjust the weightings upon receiving positive feedback for the similarity of
        anime1 with anime2.
        Preconditions:
            - anime1 in self._anime.values() and anime2 in self._anime.values()
        """
        anime1.adjust_positive_feedback(anime2)

    def _adjust_negative_feedback(self, anime1: Anime, anime2: Anime) -> None:
        """Readjust the weightings upon receiving negative feedback for the similarity of
        anime1 with anime2.
        Preconditions:
            - anime1 in self._anime.values() and anime2 in self._anime.values()
        """
        anime1.adjust_negative_feedback(anime2)

    def adjust_weighting_v1(self, anime: Anime, tag: str, reaction: str = 'upvote') -> None:
        """
        Note: this is a very inefficient operation.
        Preconditions:
            - reaction in {'upvote', 'downvote'} # decide on the name later
            - anime in self._anime
        """
        if reaction == 'upvote':
            anime.set_tag_weighting(tag, anime.get_tags()[tag] * 1.1)
        else:
            anime.set_tag_weighting(tag, anime.get_tags()[tag] * 0.9)

    def adjust_weighting_v2(self, anime1: Anime, anime2: Anime, reaction: str = 'upvote') -> None:
        """
        Preconditions:
            - reaction in {'upvote', 'downvote'}
        """
        if reaction == 'upvote':
            self._adjust_positive_feedback(anime1, anime2)
        else:
            self._adjust_negative_feedback(anime1, anime2)

    def calculate_neighbours(self, anime_name: str) -> None:
        """Add the neighbours for each anime.
        Warning: this method uses heavy computation and initializes the edges between anime.
        It is not meant to be accessible when the user's session is ongoing, and should only be
        used when the user quits their session.
        Preconditions:
            - anime in self._anime
        """
        anime_list = list(self._anime.values())
        anime = self._anime[anime_name]

        anime_list.sort(key=anime.calculate_similarity, reverse=True)
        anime.neighbours = anime_list[:NEIGHBOUR_LIMIT]
        self._anime[anime_name] = anime

    def serialize(self, output_file: str) -> None:
        """Save the neighbours of each Anime in this graph into an output file
        """
        neighbours = {}

        for anime_name in self._anime:
            neighbours[anime_name] = {
                'title': self._anime[anime_name].title,
                'neighbours': [],
                'url': self._anime[anime_name].url,
                'thumbnail': self._anime[anime_name].thumbnail,
                'detail': self._anime[anime_name].detail,
                'tags': self._anime[anime_name].get_tags()
            }
            for neighbour in self._anime[anime_name].neighbours:
                neighbours[anime_name]['neighbours'].append(neighbour.title)

        with open(output_file, 'w') as new_file:
            json.dump(neighbours, new_file)

    def prediction(self, past_choices: list[tuple[Anime, Anime]], options: list[Anime],
                   curr_anime: Anime) -> list[Anime]:
        """
        Return a prediction of which anime in options the user is likely to choose, given all
        previous choices made and the current anime.
        """
        if past_choices == []:
            return options
        else:
            prediction_weights = self._get_prediction_weights(curr_anime, past_choices)
            return sorted(options,
                          key=lambda anime: anime.prediction_similarity(prediction_weights),
                          reverse=True)

    def _get_prediction_weights(self, curr_anime: Anime, past_choices: list[tuple[Anime, Anime]]) \
            -> dict[str, float]:
        """Get the weightings required to make the predictions"""
        prediction_weights = {tag: 1 for tag in curr_anime.get_tags()}
        for pair in past_choices:
            for tag in curr_anime.get_tags():
                if tag in pair[0].get_tags() and tag in pair[1].get_tags():
                    prediction_weights[tag] += 1
        return prediction_weights

    def get_related_anime(self, anime_title: str, limit: int = 5) -> list[Anime]:
        """Return a list of up to <limit> anime that are related to the given anime,
        ordered by their similarity in descending order.
        The similarity is explained in the project report and in the Anime.py file.
        Raise ValueError if anime_title is not in the graph.
        """
        if anime_title in self._anime:
            anime = self._anime[anime_title]
            if len(anime.neighbours) > limit:
                return anime.neighbours[:limit]
            else:
                return anime.neighbours
        else:
            raise ValueError

    def add_connection(self, graph: nx.Graph(), cur_anime_title: str, det_anime_title: str) -> None:
        """Add one edge to a given graph
        Preconditions:
            - cur_anime_tile in self._anime
            - det_anime_tile not in self._anime
        """
        if cur_anime_title != det_anime_title:
            graph.add_node(det_anime_title, kind=str)
            graph.add_edge(cur_anime_title, det_anime_title)

    def _get_all_edges_pos(self, graph: nx.Graph, nxg: dict):
        """Get all edges position in networkx graph and return a tuple of edges position in x-y
        dimension
        """
        x_edge_pos = []
        y_edge_pos = []
        x_mid_pos = []
        y_mid_pos = []
        edge_similarity = []
        for edge in graph.edges:
            print([edge[0], edge[1]])

            x0, y0 = nxg[edge[0]][0], nxg[edge[0]][1]

            x1, y1 = nxg[edge[1]][0], nxg[edge[1]][1]

            x_edge_pos.extend([x0, x1, None])
            y_edge_pos.extend([y0, y1, None])

            x_mid_pos.extend([(x0 + x1) / 2, None])
            y_mid_pos.extend([(y0 + y1) / 2, None])

            edge_sim = self._anime[edge[1]].calculate_similarity(self._anime[edge[0]])

            # TODO Need to be checked, this is real time calculation and needs to consider load static similarity data
            edge_similarity.extend(
                [f'Similarity score: {edge_sim} Between {edge[0]} and {edge[1]}', None])

        return (x_edge_pos, y_edge_pos, (x_mid_pos, y_mid_pos), edge_similarity)

    def draw_graph(self, anime_title: str, depth: int, limit: int) -> plotly.graph_objs.Figure():
        """Draw a plotly graph centered around the given anime title
        Preconditions,:
            - depth <= 5 # This will be handled by the slider on the website
        """
        print([anime_title, depth, limit])

        graph = nx.Graph()
        shell = [[anime_title], []]  # [[center of graph], [other nodes]]
        queue = [(anime_title, 0)]  # title, depth
        while len(queue) != 0:
            cur = queue[0]
            shell[1].append(cur[0])
            queue.pop(0)
            print(cur[0])
            if cur[1] < depth:
                for i in self.get_related_anime(cur[0], limit):
                    self.add_connection(graph, cur[0], i.title)
                    Q.append((i.title, cur[1] + 1))

        print(shell[0], shell[1])
        print(f"total node number: {len(shell[1])}")

        if 1 + limit ** depth > 3 and len(shell[1]) >= 2:
            # nxg = nx.drawing.nx_agraph.graphviz_layout(graph, prog="twopi", root=shell[0][0])
            nxg = nx.drawing.layout.shell_layout(graph, shell)
        else:
            nxg = nx.drawing.layout.spring_layout(graph)
            print(nxg[anime_title])

        x_node_pos = [nxg[key][0] for key in graph.nodes if key != anime_title]
        y_node_pos = [nxg[key][1] for key in graph.nodes if key != anime_title]
        node_hover = [key for key in graph.nodes if key != anime_title]

        x_edge_pos, y_edge_pos, mid_pos, similarity = self._get_all_edges_pos(graph, nxg)

        all_traces = []

        central_node_trace = plotly.graph_objs.Scatter(
            x=[nxg[anime_title][0]],
            y=[nxg[anime_title][1]],
            hovertext=anime_title,
            mode="markers",
            name="nodes",
            marker={'size': 50, 'color': 'Red'}
        )

        all_traces.append(central_node_trace)

        nodes_trace = plotly.graph_objs.Scatter(
            x=x_node_pos,
            y=y_node_pos,
            hovertext=node_hover,
            mode="markers",
            name="nodes",
            marker={'size': 50, 'color': 'LightSkyBlue'}
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
            mode='markers',
            hoverinfo="text",
            marker={'size': 5, 'color': 'Black'}
        )

        all_traces.append(hover_trace)

        graph_layout = plotly.graph_objs.Layout(
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False}
        )

        figure = plotly.graph_objs.Figure(data=all_traces, layout=graph_layout)

        print("updated")

        return figure


def load_anime_graph(file_name: str) -> Graph:
    """Return the anime graph corresponding to the given dataset
    WARNING: this function will roughly take one hour to run on the full dataset.
    Preconditions:
        - file_name is the path to a json file corresponding to the anime data
          format described in the project report
    """
    anime_graph = Graph()

    with open(file_name) as json_file:
        data = json.load(json_file)
        for title in data:
            anime_graph.add_anime(title, data[title])

    count = 0
    for anime_name in anime_graph.get_all_anime():
        anime_graph.calculate_neighbours(anime_name)
        count += 1
        print(f"done {count}")

    return anime_graph


def load_from_serialized_data(file_name: str) -> Graph:
    """Return the anime graph corresponding to the given serialized dataset
        Preconditions:
            - file_name is the path to a json file corresponding to the anime data
            format described in the project report
    """
    anime_graph = Graph()

    # count = 0
    with open(file_name) as json_file:
        data = json.load(json_file)
        for title in data:
            anime_graph.add_anime(title, data[title])

        for title in data:
            anime_graph._anime[title].neighbours = [anime_graph._anime[i] for i in
                                                    data[title]['neighbours']]
            # count += 1
            # print(f"done {count}")

    return anime_graph


class FastGraph:
    """TODO: finish docstring"""
    _anime: list[Anime]

    def __init__(self) -> None:
        self._anime = list()

    def _calculate_neighbours(self, anime: Anime) -> Anime:
        """Add the neighbours for each anime.
        Warning: this method uses heavy computation and initializes the edges between anime.
        It is not meant to be accessible when the user's session is ongoing, and should only be
        used when the user quits their session.
        Preconditions:
            - anime in self._anime
        """
        anime_list = self._anime.copy()
        anime_list.sort(key=anime.calculate_similarity, reverse=True)
        anime.neighbours = anime_list[:NEIGHBOUR_LIMIT]
        return anime

    def calc_graph(self, anime_dic: dict[str, Anime]) -> dict[str, Anime]:
        """ input is a uncalculated anime graph
        return a calculated anime graph
        """
        self._anime, anime_list = list(anime_dic.values()), list(anime_dic.values())
        p = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        res = p.map(self._calculate_neighbours, anime_list)
        p.close()
        p.join()

        return {anime_list[i].title: res[i] for i in range(0, len(anime_list))}

    def load_anime_graph_multiprocess(self, file_name: str) -> Graph:
        """Return the anime graph corresponding to the given dataset
        On the full dataset, it takes about 17s using 3900x.
        Preconditions:
            - file_name is the path to a json file corresponding to the anime data
            format described in the project report
        """
        anime_graph = Graph()

        with open(file_name) as json_file:
            data = json.load(json_file)
            for title in data:
                anime_graph.add_anime(title, data[title])

        anime_graph._anime = self.calc_graph(anime_graph._anime)
        return anime_graph


if __name__ == '__main__':
    # t = time.process_time()
    # G = FastGraph().load_anime_graph_multiprocess('data/full.json')
    # G.serialize('data/full_graph.json')
    # elapsed_time = time.process_time() - t
    # print(f'process takes {elapsed_time} sec')

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E9999'],
        # 'extra-imports': ['csv', 'networkx'],
        # 'allowed-io': ['load_review_graph'],
        'max-nested-blocks': 4
    })
