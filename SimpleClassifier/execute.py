from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier
import numpy as np
import pandas as pd
from config.cfg import cfg
import plotly.graph_objects as go
import copy

dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])
gt = dataset['class']