import plotly
from anime import Anime

class Graph:
    _animes: dict[str, Anime]

    def __init__():
        ...
    
    def get_related_anime(anime_title: str, limit: int = 20) -> list[Anime]:
        ...


    def draw_graph(anime_title: str, depth: int) -> plotly.graph_objs.Figure():
        """The depth should be sent from the 
        
        Preconditions:
            - depth <= 5 # This will be handled by the slider on the website
        """
        ...

    # When we receive an upvote/downvote from the website
    def adjust_weighting(tag: str, reaction: str = 'upvote') -> None:
        """
        Preconditions:
            - reaction in {'upvote', 'downvote'} # decide on the name later
        """
        # call the adjust_weighting function in Anime

    









