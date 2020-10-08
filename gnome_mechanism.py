import numpy as np
import operator

'''
Object for running an instance of an Anish scored competition
Initialized with number of agents to participate in competition
'''
class GnomeInstance:
    def __init__(self, agent_count):
        self.agent_count = agent_count
        self.agents = np.arange(0, self.agent_count)
        self.agent_wins = {i: 0 for i in range(agent_count)}
        self.inverse_agent_count = 1 / float(agent_count)
        self.inverse_agent_count_off_one = 1 / float(agent_count - 1)

    '''
    Method for calculating quadratic score.
    '''
    def get_quadratic_score(self, y, x):
        score = 1.0 - np.power((y - x), 2)
        return score

    '''
    Method for calculating the ELF probability score of an agent being selected
    as the event winner.
    Returns the probability in [0, 1]
    '''
    def get_elf_score(self, agent_quadratic_score, total_quadratic_score):
        # All agents start with 1/n score that is tuned by ELF up or down.
        score = self.inverse_agent_count
        quadratic_score_no_agent = total_quadratic_score - agent_quadratic_score
        adj = self.inverse_agent_count_off_one * quadratic_score_no_agent
        adjustment = self.inverse_agent_count * (agent_quadratic_score - adj)

        score += adjustment

        return score


    '''
    Method for running mechanism over a single event where truth is the true
    value, and reports is a list of agent predictions.
    '''
    def run_event(self, reports, truth):
        # Calculate total quadratic score over all agents
        total_quadratic_score = 0.0
        quadratic_scores = []

        for report in reports:
            quadratic_score = self.get_quadratic_score(report, truth)
            quadratic_scores.append(quadratic_score)
            total_quadratic_score += quadratic_score

        # List to build probability distribution over agents
        probabilities = []

        for agent in range(self.agent_count):
            score = quadratic_scores[agent]
            probability = score #self.get_elf_score(score, total_quadratic_score)
            # either 0 or 1 points get added to agent score
            score_update = np.random.binomial(1,probability)
            if score_update > 0:
                self.agent_wins[agent] += score_update


    '''
    Gets the argmax over the agent wins returning the index of agent with
    the most event wins. Breaking ties uniformly
    '''
    def get_competition_winner(self):
        max_wins = -1
        bucketed_wins = {}
        for agent in self.agent_wins:
            win_number = self.agent_wins[agent]
            if win_number not in bucketed_wins:
                bucketed_wins[win_number] = [agent]
            else:
                bucketed_wins[win_number].append(agent)

            if win_number > max_wins:
                max_wins = win_number

        highest_winners = bucketed_wins[max_wins]
        winner = np.random.choice(a=highest_winners)

        return winner


if __name__ == '__main__':
    # Simple exercise of methods for debugging purposes
    # num_agents and event_count can be tuned to any value
    num_agents = 5
    event_count = 100

    gnome_mechanism = GnomeInstance(num_agents)

    for i in range(event_count):
        reports = np.zeros(num_agents)
        reports[0] = 1
        truth = 1

        gnome_mechanism.run_event(reports, truth)

    w = gnome_mechanism.get_competition_winner()
    print("Agent #{0} wins.".format(w))
