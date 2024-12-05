import sys
import copy
import torch
import random
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

class BoardPiece:

    def __init__(self, name, code, pos):
        self.name = name #name of the piece
        self.code = code #an ASCII character to display on the board
        self.pos = pos #2-tuple e.g. (1,4)

class BoardMask:

    def __init__(self, name, mask, code):
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self): #returns tuple of arrays
        return np.nonzero(self.mask)

def zip_positions2d(positions): #positions is tuple of two arrays
    x,y = positions
    return list(zip(x,y))

class GridBoard:

    def __init__(self, size=4):
        self.size = size #Board dimensions, e.g. 4 x 4
        self.components = {} #name : board piece
        self.masks = {}

    def addPiece(self, name, code, pos=(0,0)):
        newPiece = BoardPiece(name, code, pos)
        self.components[name] = newPiece

    #basically a set of boundary elements
    def addMask(self, name, mask, code):
        #mask is a 2D-numpy array with 1s where the boundary elements are
        newMask = BoardMask(name, mask, code)
        self.masks[name] = newMask

    def movePiece(self, name, pos):
        move = True
        for _, mask in self.masks.items():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
        if move:
            self.components[name].pos = pos

    def delPiece(self, name):
        del self.components['name']

    def render(self):
        dtype = '<U2'
        displ_board = np.zeros((self.size, self.size), dtype=dtype)
        displ_board[:] = ' '

        for name, piece in self.components.items():
            displ_board[piece.pos] = piece.code

        for name, mask in self.masks.items():
            displ_board[mask.get_positions()] = mask.code

        return displ_board

    def render_np(self):
        num_pieces = len(self.components) + len(self.masks)
        displ_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for name, piece in self.components.items():
            pos = (layer,) + piece.pos
            displ_board[pos] = 1
            layer += 1

        for name, mask in self.masks.items():
            x,y = self.masks['boundary'].get_positions()
            z = np.repeat(layer,len(x))
            a = (z,x,y)
            displ_board[a] = 1
            layer += 1
        return displ_board

def addTuple(a,b):
    return tuple([sum(x) for x in zip(a,b)])



class Gridworld:

    def __init__(self, size=4, mode='static'):
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GridBoard(size=4)

        #Add pieces, positions will be updated later
        self.board.addPiece('Player','P',(0,0))
        self.board.addPiece('Goal','+',(1,0))
        self.board.addPiece('Pit','-',(2,0))
        self.board.addPiece('Wall','W',(3,0))

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    #Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):
        #Setup static pieces
        self.board.components['Player'].pos = (0,3) #Row, Column
        self.board.components['Goal'].pos = (0,0)
        self.board.components['Pit'].pos = (0,1)
        self.board.components['Wall'].pos = (1,1)

    #Check if board is initialized appropriately (no overlapping pieces)
    #also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = self.board.components['Player']
        goal = self.board.components['Goal']
        wall = self.board.components['Wall']
        pit = self.board.components['Pit']

        all_positions = [piece for name,piece in self.board.components.items()]
        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0,0),(0,self.board.size), (self.board.size,0), (self.board.size,self.board.size)]
        #if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_move_go = [self.validateMove('Goal', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                #print(self.display())
                #print("Invalid board. Re-initializing...")
                valid = False

        return valid

    #Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        #height x width x depth (number of pieces)
        self.initGridStatic()
        #place player
        self.board.components['Player'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    #Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        #height x width x depth (number of pieces)
        self.board.components['Player'].pos = randPair(0,self.board.size)
        self.board.components['Goal'].pos = randPair(0,self.board.size)
        self.board.components['Pit'].pos = randPair(0,self.board.size)
        self.board.components['Wall'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def validateMove(self, piece, addpos=(0,0)):
        outcome = 0 #0 is valid, 1 invalid, 2 lost game
        pit = self.board.components['Pit'].pos
        wall = self.board.components['Wall'].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        if new_pos == wall:
            outcome = 1 #block move, player can't move to wall
        elif max(new_pos) > (self.board.size-1):    #if outside bounds of board
            outcome = 1
        elif min(new_pos) < 0: #if outside bounds
            outcome = 1
        elif new_pos == pit:
            outcome = 2

        return outcome

    def makeMove(self, action):
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0,2]:
                new_pos = addTuple(self.board.components['Player'].pos, addpos)
                self.board.movePiece('Player', new_pos)

        if action == 'u': #up
            checkMove((-1,0))
        elif action == 'd': #down
            checkMove((1,0))
        elif action == 'l': #left
            checkMove((0,-1))
        elif action == 'r': #right
            checkMove((0,1))
        else:
            pass

    def reward(self):
        if (self.board.components['Player'].pos == self.board.components['Pit'].pos):
            return -10
        elif (self.board.components['Player'].pos == self.board.components['Goal'].pos):
            return 10
        else:
            return -1

    def display(self):
        return self.board.render()
    

l1 = 64
l2 = 150
l3 = 100
l4 = 4

sync_freq = 50  # Synchronizes the frequency parameter; every 50 steps we will copy the parameters of model into model2
learning_rate = 1e-3

gamma = 0.9
epsilon = 1.0

action_set = {
    0: "u",
    1: "d",
    2: "l",
    3: "r"
}

epochs = 5000
losses = []  # Creates a list to store loss values so we can plot the trend later

mem_size = 1000  # Sets the total size of the experience relay memory
batch_size = 200  # Sets the mini-batch size
replay = deque(maxlen=mem_size)  # Creates a memory replay as a deque list
max_moves = 50  # Sets the maximum number of moves before the game is over
h = 0
sync_freq = 500  # Sets the update frequency for synchronizing the target model parameters to the main DQN
j = 0



model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
).to(device)

model2 = model2 = copy.deepcopy(model)  # Creates a second model by making an identical copy of the original Q-network model
model2.load_state_dict(model.state_dict())  # Copies the parameters of the original model

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for i in range(epochs):  # The main training loop

    game = Gridworld(size=4, mode="random")  # For each epoch, we start a new game.
    state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0  # After we create the game, we extract the state information and add a small amount of noise.
    state1 = torch.from_numpy(state1_).float().to(device)  # Converts the numpy array into a PyTorch tensor and then into a PyTorch variable
    status = 1  # Uses the status variable to keep track of whether or not the game is still in progress.
    mov = 0

    while(status == 1):  # While this game is still in progress, plays to completion and then starts a new epoch

        j += 1
        mov += 1
        qval = model(state1).to("cpu")  # Runs the Q-network forward to get its predicted Q values for all the actions
        qval_ = qval.data.numpy()

        ##########################################
        # Epsilon-Greedy Action Selection Strategy
        if (random.random() < epsilon):
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
        ##########################################

        action = action_set[action_]  # Translates the numerical action into one of the action characters that our Gridworld game expects
        game.makeMove(action)  # After selecting an action, takes the action
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state2 = torch.from_numpy(state2_).float().to(device)  # After making a move, gets the new state of the game
        reward = game.reward()
        done = True if reward > 0 else False
        exp = (state1, action_, reward, state2, done)  # Creates an experience of state, reward, action, and the next state as tuple
        replay.append(exp)  # Adds the experience to the experience replay list
        state1 = state2

        if len(replay) > batch_size:  # If the replay list isat least as long as the mini-batch size, begins the mini-batch training

            minibatch = random.sample(replay, batch_size)  # Randomly samples a subset of the replay list

            ##################################################################################
            # Separates out the components of each experience into separate mini-batch tensors
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch]).to(device)
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch]).to(device)
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch]).to(device)
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch]).to(device)
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch]).to(device)
            ##################################################################################

            Q1 = model(state1_batch)  # Recomputes Q values for the mini-batch os states to get gradients
            with torch.no_grad():  # By using the torch.no_grad() context, we tell PyTorch to not create a computational graph for the code within the context; this will save memory when we don’t need the computational graph.
                Q2 = model2(state2_batch)  # Uses the target network to get the maximum Q value for the next state

            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])  # Computes the target Q values we want the DQN to learn
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())

#             print(i, loss.item())
#             display.clear_output(wait=True)

            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if j % sync_freq == 0:  # Copies the main model parameters to the target network
                model2.load_state_dict(model.state_dict())

        if reward != -1 or mov > max_moves:  # If reward is -1, the game has not been won or lost and is still in progress
            status = 0
            mov = 0

    if epsilon > 0.1:  # Decrements the epsilon value each epoch
        epsilon -= (1 / epochs)



def test_model(model, mode="static", display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float().to(device)
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    while (status == 1):
        qval = model(state).to("cpu")
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)  # Takes the action with the highest Q value
        action = action_set[action_]
        if display:
            print("Move #: %s; Taking action: %s" % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float().to(device)
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game won! Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break
    win = True if status == 2 else False
    return win

test_model(model, 'static')