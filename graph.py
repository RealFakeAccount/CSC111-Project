"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.
Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""

from typing import Union
import json
import multiprocessing
import plotly
import networkx as nx
from anime import Anime, NEIGHBOUR_LIMIT
import parse
import os

MAX_HISTORY_LIMIT = 10


class Graph:
    """A graph of anime to represent the popularity and similarity network

    Private Instance Attributes:
        - _anime: A collection of the anime contained in this graph
        - _feedback: A list of tuples containing users' feedback
    """
    _anime: dict[str, Anime]
    _feedback: list[tuple[Anime, Anime, str]]

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
        """Return a list of all the anime in this graph
        """
        return list(self._anime)

    def get_anime_description(self, title: str) -> str:
        """Return the name of one anime.
        Return an empty string if not found
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
        """Insert anime2 into anime1.neighbours based on the similarity of anime2 with anime1
        """
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
            anime.set_tag_weighting(tag, anime.get_tag_weight(tag) * 1.1)
        else:
            anime.set_tag_weighting(tag, anime.get_tag_weight(tag) * 0.9)

    def adjust_weighting(self, anime1: Anime, anime2: Anime, reaction: str = 'upvote') -> None:
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
                'tags': {tag: self._anime[anime_name].get_tag_weight(tag) for tag in
                         self._anime[anime_name].get_all_tags()}
            }
            for neighbour in self._anime[anime_name].neighbours:
                neighbours[anime_name]['neighbours'].append(neighbour.title)

        with open(output_file, 'w') as new_file:
            json.dump(neighbours, new_file)

    def prediction(self, past_choices: list[list[Anime]], options: list[Anime],
                   curr_anime: Anime) -> list[Anime]:
        """
        Return a prediction of which anime in options the user is likely to choose, given all
        previous choices made and the current anime.

        Preconditions:
            - all(len(lst) == 2 for lst in past_choices)
        """
        if past_choices == []:
            return options
        else:
            prediction_weights = self._get_prediction_weights(curr_anime, past_choices)
            return sorted(options,
                          key=lambda anime: anime.prediction_similarity(prediction_weights),
                          reverse=True)

    def store_history(self, curr_anime: Anime, rec_anime: Anime, store_file: str) -> None:
        """Store which anime the user searched for (curr_anime), and which anime they visited out
        of the recommendations (rec_anime) in the json file store_file.

        Preconditions:
            - curr_anime in self._anime and rec_anime in self._graph
            - store_file is a json file storing a single list argument
        """
        if not os.path.exists(store_file):
            with open(store_file, 'a+') as json_file:
                data = [curr_anime, rec_anime]
                json.dump([data], json_file)
        else:
            with open(store_file, 'a+') as json_file:
                data = json.load(json_file)
                if len(data) == 10:
                    data.pop(0)
                data.append([curr_anime, rec_anime])
                json.dump(data, json_file)

    def _get_prediction_weights(self, curr_anime: Anime, past_choices: list[list[Anime]]) \
            -> dict[str, float]:
        """Get the weightings required to make the predictions

        For each list lst in past_choices, lst[0] denotes an anime that the user searched for, while
        lst[1] denotes the anime that the user clicked on next.

        Preconditions:
            - all(len(lst) == 2 for lst in past_choices)
        """
        prediction_weights = {tag: 1 for tag in curr_anime.get_all_tags()}
        for pair in past_choices:
            for tag in curr_anime.get_all_tags():
                if tag in pair[0].get_all_tags() and tag in pair[1].get_all_tags():
                    prediction_weights[tag] += 1
        return prediction_weights

    def get_related_anime(self, anime_title: str, limit: int = 5, visited: set = set()) -> list[Anime]:
        """Return a list of up to <limit> anime that are related to the given anime,
        ordered by their similarity in descending order.
        The similarity is explained in the project report and in the Anime.py file.
        Raise ValueError if anime_title is not in the graph.
        """
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
            x0, y0 = nxg[edge[0]][0], nxg[edge[0]][1]

            x1, y1 = nxg[edge[1]][0], nxg[edge[1]][1]

            x_edge_pos.extend([x0, x1, None])
            y_edge_pos.extend([y0, y1, None])

            x_mid_pos.extend([(x0 + x1) / 2, None])
            y_mid_pos.extend([(y0 + y1) / 2, None])

            edge_sim = self._anime[edge[0]].calculate_similarity(self._anime[edge[1]])

            edge_similarity.extend(
                [f'Similarity Score: {edge_sim} Between {edge[0]} and {edge[1]}', None])
            # TODO Need to be checked, this is real time calculation and needs to consider load
            #  static similarity data

        return (x_edge_pos, y_edge_pos, (x_mid_pos, y_mid_pos), edge_similarity)

    def draw_graph(self, anime_title: str, depth: int, limit: int) -> plotly.graph_objs.Figure():
        """Draw a plotly graph centered around the given anime title
        Preconditions,:
            - depth <= 5 # This will be handled by the slider on the website
        """
        graph_layout = plotly.graph_objs.Layout(
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            height=1080
        )

        visited = set(anime_title)
        graph = nx.Graph()
        shell = [[anime_title], []]  # [[center of graph], [other nodes]]
        queue = [(anime_title, 0)]  # title, depth
        while len(queue) != 0:
            cur = queue[0]
            queue.pop(0)

            for i in self.get_related_anime(cur[0], limit=limit, visited=visited):
                if i.title in visited: continue
                visited.add(i.title)

                shell[1].append(i.title)
                self.add_connection(graph, cur[0], i.title)
                if cur[1] < depth - 1:
                    queue.append((i.title, cur[1] + 1))

        print(shell[0], shell[1])
        print(f"total node number: {len(shell[1])}")

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

        print(f"key: {[key for key in graph.nodes]}")

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

        print("updated")

        return figure

    def get_anime(self) -> dict[str, Anime]:
        """Return the _anime attribute of the graph"""
        return self._anime

    def store_feedback(self, reaction: str, curr_anime: Anime, feedback_anime: Anime) -> None:
        """
        Store the user's feedback
        Preconditions:
            - curr_anime in self._anime
            - feedback_anime in self._anime
            - reaction in {'upvote', 'downvote'}
        """
        self._feedback.append((curr_anime, feedback_anime, reaction))

    def implement_feedback(self) -> None:
        """
        Mutate the anime in self._anime according to the feedback received.
        This method also mutates self._feedback so that, by the time this function has finished
        executing, self._feedback is empty.
        WARNING: This method also recomputes the entire graph, so it takes as long as the
        initialization of the graph with data.
        """
        if self._feedback != []:
            length = len(self._feedback)

            for _ in range(length):
                anime1, anime2, response = self._feedback.pop(0)
                self.adjust_weighting(anime1, anime2, response)

        assert self._feedback == []

        # need to recompute neighbours and edges
        self._anime = LoadGraphFast().calc_graph(self._anime)


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


class LoadGraphFast:
    """A data class meant for loading the graph faster than before."""
    _anime: list[Anime]

    def __init__(self) -> None:
        """Initialize ..."""
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
        WRNING: This may absolutely wreck your device. For context, on the full dataset, it takes
        about 17s using 3900x.
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


if __name__ == "__main__":
    # import time
    # t = time.process_time()
    # G = load_from_serialized_data("data/full_graph.json")
    # print(sum(len(i._tags) == 0 for i in G._anime.values()))
    # elapsed_time = time.process_time() - t
    # print(f"process takes {elapsed_time} sec")
    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E9999', 'E9998'],
        'max-nested-blocks': 4
    })
