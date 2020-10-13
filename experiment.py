import numpy as np
import argparse
import matplotlib.pyplot as plt

import elf_mechanism

class Experiment:
    def __init__(self, agent_count):
        self.agent_count = agent_count

    '''
    Run n events with the same truth value and the same reports for all agents
    where agent #0 is the predestined winner, and all others are off by gap.
    Returns proportion of competitions agent #0 wins.

    Note: Gap should be given relative to truth:
    truth = 1, gap = -0.5, naive_report = 0.5
    truth = 0.5, gap = 0.1 naive_report = 0.6
    '''
    def run_m_events_uniform_gap(self, epochs, m, truth, gap, competition):
        naive_report = truth + gap
        reports = np.ones(self.agent_count) * naive_report

        reports[0] = truth
        WINNER = 0

        success_count = 0
        for _ in range(epochs):
            for __ in range(m):
                competition.run_event(reports=reports, truth=truth)

            winner = competition.get_competition_winner()
            if winner == WINNER:
                success_count += 1

        success_proportion = success_count / float(epochs)
        return success_proportion

    def experiment_varying_m_uniform_gap(self, m_range, truth, gap, experiment):
        data = []
        epochs = 100
        for m in m_range:
            print(m)
            proportion = experiment(epochs, m, truth, gap)
            data.append(proportion)

        title = "Proportion of ELF Competitions True Best Model Chosen"
        self.plot_results(m_range, data, title)

    def plot_results(self, x, y, title):
        plt.scatter(x, y)
        plt.plot(x, y)
        plt.ylim(0,1)
        plt.xlim(0)
        plt.xlabel('Number of Events')
        plt.ylabel('Proportion')
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    agent_count = 10
    experiment = Experiment(agent_count=agent_count)
    m_range = np.array(range(100,5001,100))
    truth = 0.5
    gap = -0.4
    method = experiment.run_m_events_uniform_gap
    competition = elf_mechanism.ElfInstance(agent_count)
    experiment.experiment_varying_m_uniform_gap(m_range, truth, gap, method, competition)
