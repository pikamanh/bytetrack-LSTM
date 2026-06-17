#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

try:
    from .fast_coco_eval_api import COCOeval_opt
except ImportError:
    from pycocotools.cocoeval import COCOeval as COCOeval_opt
