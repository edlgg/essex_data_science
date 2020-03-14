# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in
# the UCTPlayGame() function at the bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import log, sqrt
import random
import csv
import shutil
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier


class OthelloState:
    """ A state of the game of Othello, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 2 = player 2 (O).
        In Othello players alternately place pieces on a square board - each piece played
        has to sandwich opponent pieces between the piece played and pieces already on the
        board. Sandwiched pieces are flipped.
        This implementation modifies the rules to allow variable sized square boards and
        terminates the game as soon as the player about to move cannot make a move (whereas
        the standard game allows for a pass move).
    """

    def __init__(self, sz=8):
        # At the root pretend the player just moved is p2 - p1 has the first move
        self.playerJustMoved = 2
        self.board = []  # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0  # size must be integral and even
        for _ in range(sz):
            self.board.append([0]*sz)
        self.board[int(sz/2)][int(sz/2)
                              ] = self.board[int(sz/2-1)][int(sz/2-1)] = 1
        self.board[int(sz/2)][int(sz/2-1)
                              ] = self.board[int(sz/2-1)][int(sz/2)] = 2

    def getFlatBoard(self):
        return sum(self.board, [])

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x, y) = (move[0], move[1])
        assert x == int(x) and y == int(y) and self.IsOnBoard(
            x, y) and self.board[x][y] == 0
        m = self.GetAllSandwichedCounters(x, y)
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[x][y] = self.playerJustMoved
        for (a, b) in m:
            self.board[a][b] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0 and self.ExistsSandwichedCounter(x, y)]

    def AdjacentToEnemy(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x+dx, y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                return True
        return False

    def AdjacentEnemyDirections(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        es = []
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x+dx, y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                es.append((dx, dy))
        return es

    def ExistsSandwichedCounter(self, x, y):
        """ Does there exist at least one counter which would be flipped if my counter was placed at (x,y)?
        """
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            if len(self.SandwichedCounters(x, y, dx, dy)) > 0:
                return True
        return False

    def GetAllSandwichedCounters(self, x, y):
        """ Is (x,y) a possible move (i.e. opponent counters are sandwiched between (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            sandwiched.extend(self.SandwichedCounters(x, y, dx, dy))
        return sandwiched

    def SandwichedCounters(self, x, y, dx, dy):
        """ Return the coordinates of all opponent counters sandwiched between (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.IsOnBoard(x, y) and self.board[x][y] == self.playerJustMoved:
            sandwiched.append((x, y))
            x += dx
            y += dy
        if self.IsOnBoard(x, y) and self.board[x][y] == 3 - self.playerJustMoved:
            return sandwiched
        else:
            return []  # nothing sandwiched

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        jmcount = len([(x, y) for x in range(self.size)
                       for y in range(self.size) if self.board[x][y] == playerjm])
        notjmcount = len([(x, y) for x in range(self.size)
                          for y in range(4) if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount:
            return 1.0
        elif notjmcount > jmcount:
            return 0.0
        else:
            return 0.5  # draw

    def __repr__(self):
        s = ""
        for y in range(self.size-1, -1, -1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        # the only part of the state that the Node needs later
        self.playerJustMoved = state.playerJustMoved

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key=lambda c: c.wins /
                   c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent+1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for _ in range(1, indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose=False, clf=None):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state=rootstate)

    for _ in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []:  # while state is non-terminal
            available_moves = state.GetMoves()
            selected_move = None
            if clf and random.random() < 0.9:
                X = state.getFlatBoard()
                X.append(state.playerJustMoved)
                move = clf.predict([X])
                move = tuple(move[0])
                if move in available_moves:
                    selected_move = move

            if not selected_move:
                selected_move = random.choice(available_moves)

            state.DoMove(selected_move)

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            # state is terminal. Update node with result from POV of node.playerJustMoved
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose):
        print(rootnode.TreeToString(0))
    # else:
    #     print(rootnode.ChildrenToString())

    # return the move that was most visited
    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move


def UCTPlayGame(clf1, clf2, itermax1=100, itermax2=100, verbose=False):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """
    state = OthelloState(6)
    samples = []  # initialices list to create csv
    winner = None
    while (state.GetMoves() != []):
        if verbose:
            print(str(state))
        if state.playerJustMoved == 1:
            # play with values for itermax and verbose = True
            m = UCT(rootstate=state, itermax=itermax1, verbose=False, clf=clf1)
        else:
            m = UCT(rootstate=state, itermax=itermax2, verbose=False, clf=clf2)
        if verbose:
            print("Best Move: " + str(m) + "\n")
        # All the state is saved in an array to store in a csv
        sample = state.getFlatBoard()
        sample.append(state.playerJustMoved)
        sample.append(m[0])
        sample.append(m[1])
        samples.append(sample)
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
        winner = state.playerJustMoved
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
        winner = 3 - state.playerJustMoved
    else:
        print("Nobody wins!")

    ###
    ###
    return winner, samples


def save_samples_to_csv(file_name, l):
    with open(file_name, "a") as f:
        writer = csv.writer(f)
        writer.writerows(l)
    return


def create_temp_directories():
    shutil.rmtree('./datasets', ignore_errors=True)
    shutil.rmtree('./classifiers', ignore_errors=True)
    os.mkdir('./datasets')
    os.mkdir('./classifiers')
    return


def train_clf(epoc):
    f = open(f"datasets/samples_{epoc}.csv")
    df = pd.read_csv(f, header=None)
    y = df.iloc[:, -2:]
    X = df.iloc[:, :-2]

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    joblib.dump(clf, f'./classifiers/clf_v{epoc}.pkl')
    return clf


def train(epocs, games_per_epoc):
    create_temp_directories()

    clf1 = None
    clf2 = None
    for epoc in range(epocs):
        for game in range(games_per_epoc):
            print('epoc')
            print(epoc)
            print('game')
            print(game)
            # Function tambien debe de regresar quien gano
            _, samples = UCTPlayGame(
                clf1, clf2, 100, 100, False)
            save_samples_to_csv(f"datasets/samples_{epoc}.csv", samples)

        clf1 = train_clf(epoc)
        clf2 = clf1
        # if epoc < 25:
        #     os.remove(f"datasets/samples_{epoc}.csv")


def compareClfs(clf1, clf2, itermax1, itermax2):
    results = {}

    for _ in range(50):
        winner, _ = UCTPlayGame(clf1, clf2, itermax1, itermax2, False)
        if winner not in results:
            results[winner] = 1
        else:
            results[winner] = results[winner] + 1

        print(results)
    return


if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players.
    """

    EPOCS = 5
    GAMES_PER_EPOC = 5
    train(EPOCS, GAMES_PER_EPOC)

    clf1 = joblib.load(f'classifiers/clf_v0.pkl')
    clf2 = joblib.load(f'classifiers/clf_v{EPOCS-1}.pkl')
    # clf2 = joblib.load(f'classifiers/clf_v1.pkl')
    # clf2 = None
    compareClfs(clf1, clf2, 200, 200)
