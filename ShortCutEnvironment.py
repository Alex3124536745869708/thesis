import random
import numpy as np

class Environment(object):

    def __init__(self):
        pass

    def reset(self):
        '''Reset the environment.
        
        Returns:
           starting_position: Starting position of the agent (the state).
        '''
        raise Exception("Must be implemented by subclass.")
    
    def render(self):
        '''Render environment to screen.'''
        raise Exception("Must be implemented by subclass.")
    
    def step(self, action):
        '''Take action.
        
        Arguments:
           action: action to take.
        
        Returns:
           reward: reward of action taken.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def possible_actions(self):
        '''Return list of possible actions in current state.
        
        Returns:
          actions: list of possible actions.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def state(self):
        '''Return current state.

        Returns:
          state: environment-specific representation of current state.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def state_size(self):
        '''Return the number of elements of the state space.

        Returns:
          state_size: number of elements of the state space.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def action_size(self):
        '''Return the number of elements of the action space.

        Returns:
          state_size: number of elements of the action space.
        '''
        raise Exception("Must be implemented by subclass.")
    
    def done(self):
        '''Return whether current episode is finished and environment should be reset.

        Returns:
          done: True if current episode is finished.
        '''
        raise Exception("Must be implemented by subclass.")

class ShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 5 # amount of rows (=column length)
        self.c = 5 # amount of columns (=row length)
        self.rng = random.Random(seed)
        # s stands for all states (the map)
        s = np.zeros((self.r, self.c+1), dtype=str)
        s[:] = 'X'
        s[:,-1] = '\n'
        s[0,0] = 'B'
        s[1,2:4] = 'B'
        s[2,1] = 'B'
        s[2,2] = 'C'
        s[2,4] = 'G'
        s[3,4] = 'B'
        s[4,2:4] = 'C'
        self.s = s
        self.reset()
    
    def reset(self):
        self.y = 3
        self.x = 1
        self.starty = self.y
        self.startx = self.x
        self.isdone = False
        return self.state()
    
    def state(self):
        return self.y*self.c + self.x
    
    def state_size(self):
        return self.c*self.r
    
    def action_size(self):
        return 4
    
    def done(self):
        return self.isdone
    
    def possible_actions(self):
        return [0, 1, 2, 3]
    
    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')
        
        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')
        # y is the hight, y+1 is down
        # x is the width, x+1 is to the right
        # r is the row (denotes the hight)
        # c is the column (denotes the width)
        prev_location = (self.y, self.x) # for walking into walls
        if action == 0:
            if self.y>0:
                self.y -= 1
        elif action == 1:
            if self.y<self.r-1:
                self.y += 1
        elif action == 2:
            if self.x>0:
                self.x -= 1
        elif action == 3:
            if self.x<self.c-1:
                self.x += 1
        
        if self.s[self.y, self.x]=='G': # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x]=='C': # Fall off cliff
                self.y = self.starty
                self.x = self.startx
                return -100
        elif self.s[self.y, self.x]== 'B': # walk into a wall (a boulder)
            self.y = prev_location[0]
            self.x = prev_location[1]
            return -1
        return -1
    
    
    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        string = "Shortcutenv:\n"
        clean_str = ''.join(map(str, s.flatten()))
        string += clean_str
        with open("environment.txt", "a", encoding="utf-8") as text_file: 
            text_file.write(string) # saving the string


class WindyShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 5 # amount of rows (=column length)
        self.c = 5 # amount of columns (=row length)
        self.rng = random.Random(seed)
        # s stands for all states (the map)
        s = np.zeros((self.r, self.c+1), dtype=str)
        s[:] = 'X'
        s[:,-1] = '\n'
        s[0,0] = 'B'
        s[1,2:4] = 'B'
        s[2,1] = 'B'
        s[2,2] = 'C'
        s[2,4] = 'G'
        s[3,4] = 'B'
        s[4,2:4] = 'C'
        self.s = s
        self.reset()
    
    def reset(self):
        self.y = 3
        self.x = 1
        self.starty = self.y
        self.startx = self.x
        self.isdone = False
        return self.state()
    
    def state(self):
        return self.y*self.c + self.x
    
    def state_size(self):
        return self.c*self.r
    
    def action_size(self):
        return 4
    
    def done(self):
        return self.isdone
    
    def possible_actions(self):
        return [0, 1, 2, 3]
    
    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')
        
        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')
        # y is the hight, y+1 is down
        # x is the width, x+1 is to the right
        # r is the row (denotes the hight)
        # c is the column (denotes the width)
        prev_location = (self.y, self.x) # for walking into walls
        if action == 0:
            if self.y>0:
                self.y -= 1
        elif action == 1:
            if self.y<self.r-1:
                self.y += 1
        elif action == 2:
            if self.x>0:
                self.x -= 1
        elif action == 3:
            if self.x<self.c-1:
                self.x += 1

        if self.rng.random()<0.5:
            # Wind! blows agent downwards
            if self.y < self.r-1:
                self.y += 1
        
        if self.s[self.y, self.x]=='G': # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x]=='C': # Fall off cliff
                self.y = self.starty
                self.x = self.startx
                return -100
        elif self.s[self.y, self.x]== 'B': # walk into a wall (a boulder)
            self.y = prev_location[0]
            self.x = prev_location[1]
            return -1
        return -1
    
    
    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        string = "WindyShortcutenv:\n"
        clean_str = ''.join(map(str, s.flatten()))
        string += clean_str
        with open("environment.txt", "a", encoding="utf-8") as text_file: 
            text_file.write(string) # saving the string
