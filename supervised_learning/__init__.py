# -*- coding: utf-8 -*-

from . import models

def _post_load_demo_data(env):
    if env.ref('base.module_supervised_learning').demo:
        env['project.task']._post_load_demo_data()