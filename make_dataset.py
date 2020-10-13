import numpy as np
import pandas as pd
import math
from experiment import Experiment
from elf_mechanism import ElfInstance
from gnome_mechanism import GnomeInstance

epochs = 100
truth = 0.5

rows = []
for n in [100, 200, 300, 400, 500]:
# for m in [100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
    for m in range(10, 110, 10):
        for epsilon in [.01, .05, .1, .2, .3]:
            gap = math.sqrt(epsilon)
            elf = ElfInstance(m)
            gnome = GnomeInstance(m)
            experiment = Experiment(m)
            elf_score = experiment.run_m_events_uniform_gap(epochs, m, truth, gap, elf)
            gnome_score = experiment.run_m_events_uniform_gap(epochs, m, truth, gap, gnome)
            point = {"m": m, "n": n, "epsilon": epsilon, "elf": elf_score, "gnome": gnome_score}
            print(point)
            rows += [point]

pd.DataFrame(rows).to_pickle("data/small_dataset.pkl")