from dataclasses import dataclass
from utils.parser_util import BaseOptions, DataOptions


@dataclass
class humanml_motion_rel(BaseOptions, DataOptions):
    dataset: str = 'humanml'
    data_dir: str = ''
    abs_3d: bool = False


@dataclass
class humanml_motion_abs(BaseOptions, DataOptions):
    dataset: str = 'humanml'
    data_dir: str = ''
    abs_3d: bool = True

@dataclass
class trumans_motion_smpl_occ(BaseOptions, DataOptions):
    dataset: str = 'trumans'
    data_dir: str = ''
    smpl: bool = True
    scene: str = 'occ'

@dataclass
class humanml_motion_proj1(humanml_motion_abs):
    use_random_proj: bool = True
    random_proj_scale: float = 1


@dataclass
class humanml_motion_proj2(humanml_motion_abs):
    use_random_proj: bool = True
    random_proj_scale: float = 2


@dataclass
class humanml_motion_proj5(humanml_motion_abs):
    use_random_proj: bool = True
    random_proj_scale: float = 5


@dataclass
class humanml_motion_proj10(humanml_motion_abs):
    use_random_proj: bool = True
    random_proj_scale: float = 10


@dataclass
class humanml_traj(humanml_motion_abs):
    traj_only: bool = True