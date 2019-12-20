import numpy as np
from tools import simple_plot, draw_learning_curve, access_learning, compare_histogram
import matplotlib.pyplot as plt


'''
with open("seeds.txt", 'r') as file:
    seeds = file.read()[1:-1].split(' ')
    for s in range(len(seeds)):
        seeds[s] = int(seeds[s])
seeds = [1042, 1209]
le_list, ole_list = [], []
for seed in seeds:
    ve, cve = np.load("save/s"+str(seed)+"/variance_explained.npy", allow_pickle=True)
    simple_plot(ve)
    simple_plot(cve)
    opl_list = np.load("save/s"+str(seed)+"/outside_manifold_perturbation_learning_curves.npy", allow_pickle=True)
    draw_learning_curve(opl_list, 200, 200, name="outside manifold")
    pl_list = np.load("save/s"+str(seed)+"/within_manifold_perturbation_learning_curves.npy", allow_pickle=True)
    draw_learning_curve(pl_list, 200, 200, name="within manifold")
    le, ole = [], []
    for p in range(25):
        le.append(access_learning(pl_list[p], 200, 200))
        ole.append(access_learning(opl_list[p], 200, 200))
    le, ole = np.array(le), np.array(ole)
    le_list.append(le)
    ole_list.append(ole)
le = np.concatenate(le_list)
ole = np.concatenate(ole_list)
compare_histogram(le[:,0], ole[:,0], xname="amount of learning", save_path="save/")
compare_histogram(le[:,-1], ole[:,-1], xname="relative learning after effect", save_path="save/")
'''

