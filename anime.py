class Anime: # a node in the graph
    tags: dict[str, float] # tag: weighting(0-1)
    title: str
    db_score: float
    detail: str # introduction to one anime
    # related_anime: list[Anime]
    
    
    def __init__():
        ...
    
    def set_weights():
        ...
   
    def calculate_similarity():
        ...

    def set_tag_weighting(tag: str, new_weighting: float):
        ...
    
    def renormalize_tags():
        ...



   
def load_from_json(file_name: str):
    ...