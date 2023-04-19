from pydantic import BaseModel, Field
from pydantic.typing import Literal


class LoadModelAndStuffConfig(BaseModel):
    model_path: str


class StitchingConfig(BaseModel):
    method: str = Literal['Quilting', 'Gradient', 'PatchBased']


class Quilting(BaseModel):
    map_size: tuple[int, int]
    patch_size: int
    overlap_part: float
    coef: int
    map_type: str = Literal['png', 'tiff', 'jpg']
    use_global_map: bool


class Gradient(BaseModel):
    map_size: tuple[int, int]
    map_type: str


class PatchBased(BaseModel):
    map_size: tuple[int, int]
    map_type: str = Literal['png', 'tiff', 'jpg']
    image_shape = tuple[int, int]
    maps_image_vector_size: int
    method: str = Literal['linear', 'gaussian', 'quilting']
    overlap_part: float = Field(gt=0, lt=1)
    

class UploadConfig(BaseModel):
    load_model_and_stuff: LoadModelAndStuffConfig
    stitching_method: StitchingConfig
    quilting: Quilting
    gradient: Gradient
    patch_based: PatchBased
