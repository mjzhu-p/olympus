#!/usr/bin/env python

from olympus.utils.misc import check_planner_module

#===============================================================================

planner = 'Dragonfly'
module  = 'dragonfly'
link    = 'https://pypi.org/project/dragonfly-opt/'
check_planner_module(planner, module, link)

#===============================================================================

from .wrapper_dragonfly import Dragonfly
