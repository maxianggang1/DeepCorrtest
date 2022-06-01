from preprocess import Preprocess
from training import Server

p = Preprocess()
p.main_run()

# we tried the training process on 4 Tesla V100 cards.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

t = Server()
t.train()
props,y_truth = t.predict()