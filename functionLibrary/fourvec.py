# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:11:30 2018

@author: g

"""

import numpy as np
import pandas as pd
import os as os
import matplotlib.pyplot as pyplot
import math as math

def invMass(df,system,p,px,py,pz):
    return (df['%s%s' % (system,p)]*df['%s%s' % (system,p)] - df['%s%s' % (system,px)]*df['%s%s' % (system,px)] - df['%s%s' % (system,py)]*df['%s%s' % (system,py)] - df['%s%s' % (system,pz)]*df['%s%s' % (system,pz)]).apply(math.sqrt)