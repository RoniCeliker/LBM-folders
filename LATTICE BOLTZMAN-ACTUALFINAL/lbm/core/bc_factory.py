from lbm.util.factory import Factory

from lbm.core.bc.orphan import Orphan
from lbm.core.bc.bounce_back import BounceBack
from lbm.core.bc.anti_bounce_back import AntiBounceBack
from lbm.core.bc.symmetry import Symmetry


bc_factory = Factory()
bc_factory.register("ORPHAN", Orphan)
bc_factory.register("BOUNCE_BACK", BounceBack)
bc_factory.register("ANTI_BOUNCE_BACK", AntiBounceBack)
bc_factory.register("SYMMETRY", Symmetry)
