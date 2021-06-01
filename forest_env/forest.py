import os, cv2, copy
import numpy as np

from gym import spaces, Env
import matplotlib.pyplot as plt

# class that defines the grid environment to be used
class Forest(Env):
 
    metadata = {'renderer.modes': []}
 
    # function that defines the action and observation space
    # outputs - None
    def __init__(self, one_hot_encoding=True):
        self.ohe = one_hot_encoding
        self.stoh = False
        # define the environment
        self.dims = (5, 5)                                                     # dimensions of the environment
        self.observation_space = spaces.Discrete(np.product(self.dims))        # define number of states
        self.action_space = spaces.Discrete(4)                                 # define number of actions
 
        # define initial states of the environment        
        self.safe_house = np.subtract(self.dims, 1)                            # goal state, the safe house! (force the safe house into the corner of the grid)
        self.init_agent = [0, 0]                                               # starting position of the agent
        self.init_energy = 25                                                  # initial energy, needed if environment needs to be reset (maximum timesteps)
        self.queue_queue = []                                                  # a queue of lines, so queue queue. get it? british joke. it's for rendering. 
        self.agent_img = 'agent_alive_up'                                      # for rendering, image of agent's last state
 
        # load images used to visualize the environment
        img_dir = 'images'
        self.images = {}
        fns = [i for i in os.listdir(img_dir) if i.endswith('.png')]
        for c, fn in enumerate(fns):
            image = cv2.imread(os.path.join(img_dir, fn), 1)
            self.images[fn.split('.')[0]] = image
            print('\rloading images.. %d/%d' % (c+1, len(fns)), end='')
        print('\rloading images.. DONE.')
    
        # define points of interest within the forest
        init_picnic_basket = POI(impact=.1,                                     # picnic basket -> GOOD. someone left a picnic basket behind with a lot of food. YAY.
                                 locations=set([(4,0), (4,3)]),
                                 img_avail='picnic_avail')
        init_berries = POI(impact=.05,                                          # berries       -> wildberries, should be good. GOOD.
                           locations=set([(2,1), (3,2), (3,4)]), 
                           img_avail='berries_avail')                                     
        init_deer_poop = POI(impact=-.05,                                       # deer poop     -> BAD. looks like berries, NOT BERRIES. NOT GOOD. 
                             locations=set([(1,2), (0,3)]), 
                             img_avail='deer_poop_avail')      
        init_twig = POI(impact=-.05,                                             # twigs         -> BAD. it's dark. he tripped. PAIN. 
                        locations=set([(1,4), (2,3), (4,1)]),
                        img_avail='twig')
        init_trees = POI(impact=-.1,                                            # trees         -> BAD. don't run into trees. not good. 
                         locations=set([(1,3), (3,0)]), 
                         img_avail='tree')                                       
        init_bear_trap = POI(impact=-1,                                         # bear trap     -> BAD. if a bear can't handle it, you can't either. YOU'RE DEAD. 
                             locations=set([(3,3)]), 
                             img_avail='bear_trap')                                  
        safe_house = POI(impact=1,
                         locations=set([tuple(self.safe_house)]),
                         img_avail='house')
 
        # dictionary of points of interest, to make things a little easier on the code
        self.POIs = {'picnic': init_picnic_basket,
                     'berries': init_berries,
                     'deer_poop': init_deer_poop,
                     'twig': init_twig, 
                     'trees': init_trees,
                     'trap': init_bear_trap,
                     'house': safe_house}
        
        # create the environment             
        self.init_background = self.images['background']
        
        self.init_state = np.zeros(self.dims)                                             # define the environment state
        for i, name in enumerate(self.POIs):                                              # for each point of interest, 
            poi = self.POIs[name]                                                             # get the POI
            for loc in poi.locations:                                                         # for each location where POI is located, 
                self.init_state[tuple(loc)] = poi.impact                                          # set the impact of the POI
                image = poi.img_avail
                self.init_background = self.__draw_poi__(image, loc, self.init_background)        # update the visualization image
 
                print('\rdrawing initial environment.. %d/%d' % (i+1, len(self.POIs)), end='')
        print('\rdrawing initial environment.. DONE.')
 
 
    # function that resets the state of the environment to the initial state
    # inputs - None
    # outputs - the initial observation state
    def reset(self):
        self.energy = self.init_energy
        self.agent = copy.copy(self.init_agent)
        self.state = copy.deepcopy(self.init_state)
 
        self.background = copy.deepcopy(self.init_background)
        self.background_n_agent = copy.deepcopy(self.background)
        self.queue_queue = []
        self.agent_img = 'agent_alive_down'
        
        observation = self.to_state(self.agent)
        if self.ohe:
            ret = np.zeros(self.observation_space.n)
            ret[observation - 1] = 1
            return ret
        else:
            return observation
 
 
    # function that executes one timestep within the environment
    # inputs - action - the action to be taken in this timestep
    # outputs - observation - the new state after an action is taken 
    #         - reward - float, the updated reward
    #         - done - True if reached the goal state, False otherwise
    #         - info - additional infomation (nothing for now)
    def step(self, action):  
        start_pos = np.array(self.agent) # save the current position for line drawing
 
        # update the position of the agent based on the action 
        self.agent, agent_img = self.__calculate_direction__(action)
        self.energy -= 1
        observation = self.to_state(self.agent)
 
        # check if agent is done
        if (self.agent == self.safe_house).all(): # agent reached safe house
            done = True
        elif self.energy <= 0: # agent ran out of energy
            agent_img = 'agent_no_energy'
            done = True
        else:
            done = False
 
        # for rendering
        self.queue_queue.append((start_pos, np.array(self.agent)))
        self.agent_img = agent_img
 
        reward = self.state[tuple(self.agent)]
        
        self.state[tuple(self.agent)] = 0
        
        info = {}

        observation = self.to_state(self.agent)
        if self.ohe:
            ret = np.zeros(self.observation_space.n)
            ret[observation - 1] = 1
            return ret, reward, done, info
        else:
            return observation, reward, done, info
 
 
    # function that, given an action, returns the next state and the probability of that state
    # inputs - loc - optional, location to explore. if not defined, will be set to agent's current position
    # outputs - dictionary key-value pair of (action -> (state, probability))
    def explore(self, state=None):
        if state is None:
            loc = self.agent
        else:
            loc = self.to_coord(state)
            
        ret = {} # dictionary of (action -> (reward, prob of reward))
        x, y = loc 
        actions = [[x, y - 1], # up
                   [x - 1, y], # left
                   [x, y + 1], # down 
                   [x + 1, y]] # right
        
        # clip resulting coordinates so no coordinates are out of bounds
        for i, _ in enumerate(actions):
            actions[i] = np.clip(actions[i], 0, self.dims[0] - 1)
            actions[i] = self.to_state(actions[i])
            
        
        for i, a in enumerate(actions):
            if not self.stoh:
                ret[i] = [(self.to_state(a), 1)]
            else:
                ret[i] = [(self.to_state(a), .5)]
                
                if i % 2 == 0:
                    others = [(actions[1], .25),
                              (actions[3], .25)]
                else:
                    others = [(actions[0], .25),
                              (actions[2], .25)]
                ret[i].extend(others)
        return ret
 
 
    # function that converts coordinate on the grid to a state number
    # inputs - (x, y), coordinate location
    # outputs - int, state number
    def to_state(self, coord):
        x, y = coord
        return int((self.dims[1]) * y + x)
 
 
    # function that converts state number to coordinate i ngrid
    # inputs - int, state number
    # outputs - [x, y], coordinate location 
    def to_coord(self, state_num):
        return [int(state_num % self.dims[0]), int(state_num // self.dims[1])]
 
 
    # function that visualizes the environment
    # inputs - None
    # outputs - None, but shows the environment
    def render(self):
        # draw the lines representing the agent's path
        while self.queue_queue:
            p, q = self.queue_queue.pop(0)
            self.__draw_line__(p, q, self.background)
            
        self.__draw_agent__(self.agent_img)
        plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
        plt.axis('off')
 
        plt.imshow(cv2.cvtColor(self.background_n_agent, cv2.COLOR_BGR2RGB))
        return self.background_n_agent
 
 
    def set_stoh(self, stohchastic=False):
        self.stoh = stohchastic
        return 
 
 
    # helper function for step(), calculates the new direction
    # inputs - action - action from Agent
    # outputs - agent - the agent position to be updated
    #         - agent_img - image that is needed for rendering 
    def __calculate_direction__(self, action):
        if self.stoh:
            roll = np.random.uniform()
            if action % 2 == 0: # moving up/down
                if roll < .25:
                    action = 1
                elif roll < .5:
                    action = 3
                
            else: # moving left/right
                if roll <.25:
                    action = 0
                elif roll < .5:
                    action = 2
 
        agent_img = None
        agent = self.agent
        # even moving vertically, odd moving horizontally 
        if action == 0:
            agent[1] -= 1
            agent_img = 'agent_alive_up'   
        elif action == 2:
            agent[1] += 1
            agent_img = 'agent_alive_down'
        elif action == 1:
            agent[0] -= 1
            agent_img = 'agent_alive_left'
        elif action == 3:
            agent[0] += 1
            agent_img = 'agent_alive_right'
        
        agent = np.clip(agent, 0, self.dims[0] - 1)
        return agent, agent_img
    
    
    # helper function for step(), updates the environment by drawing the agent's new location
    # inputs - agent_img - image to be drawn
    # outputs - None
    def __draw_agent__(self, agent_img):  
        temp = np.copy(self.background)
        temp = self.__draw_poi__(agent_img, self.agent, temp)
        self.background_n_agent = temp
        return 
    
    
    # helper function for step(), draws a line onto an image
    # inputs - pc - tuple(x, y), starting point of the line
    #        - qc - tuple(x, y), ending point of the line
    #        - image - image that the line is to be drawn on
    # outputs - image with line drawn in specified coordinates
    def __draw_line__(self, pc, qc, image):
        width, height = np.divide(image.shape[:2], self.dims)
        mpw, mph = width/2, height/2
        sp = (int(pc[0] * width + mpw), int(pc[1] * height + mph))
        ep = (int(qc[0] * width + mpw), int(qc[1] * height + mph))
        
        cv2.arrowedLine(image, sp, ep, color=(0, 255, 0), thickness=5)
        return image
    
 
    # helper function for step(), draws an image onto another image (aka POI onto background)
    # inputs - image - string, name of the image that is to be drawn
    #        - coords - (x, y), coordinates where the image is to be drawn 
    #        - background - image, background environment
    # outputs - the updated background (inluding image at the given coordinate)
    def __draw_poi__(self, name, coords, background):
        image = self.images[name]
        width, height = [int(a) for a in np.divide(background.shape[:2], self.dims)]
        for w in range(width):
            for h in range(height):
                i = coords[0] * width + w
                j = coords[1] * height + h
                if (image[h][w] != [255, 255, 255]).all():
                    background[j][i] = image[h][w]
        return background
 
 
class POI:
    # class that defines a point of interest (aka rewards, negative and positive) within the environment
 
    # function that initializes a POI class
    # inputs - impact - integer, the impact it will have on the agent, can be a negative or positive impact
    #        - locations - set of [x, y], locations of the POIs
    #        - img_avail - string, name of the image when the object is available 
    
    # outputs - None
    def __init__(self, impact=0, locations=[], img_avail=None):
        self.impact = impact
        self.locations = locations
        self.img_avail = img_avail