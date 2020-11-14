try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

import random
import torch
import numpy as np

from speed_estimator import SpeedEstimator

class ExpertAgent(Agent):

    def get_position(self, state, channel):
        # idx = torch.nonzero(state[channel])

        x, y = np.nonzero(state[channel,:,:])

        assert len(x) == 1
        return (x[0],y[0])

    def find_max_speed_no_crash(self, cars, lane, y):

        # our estimator will tell us the minimum speed the lane's cars are travelling
        max_speed = self.speed_estimator.min_speed[lane]

        # have empty space in front
        if y >= 1 and cars[lane, y-1] == 0:
            max_speed -= 1
            # have empty space 2 spaces in front
            if y >= 2 and cars[lane, y-2] == 0:
                max_speed -= 1

        # check if the agent is getting impossible regions by going forward too fast
        if y + max_speed < lane:
            max_speed = min(lane - y, -1)

        return max(max_speed, self.agent_max_speed)

    def up_lane_danger_level(self, cars, lane, y):
        assert lane > 0
        
        # have a car right above
        if cars[lane-1, y]:
            return 1

        # if the right-back has a car, the mild risk of collision
        if cars[lane-1, (y+1)%self.width]:
            return self.speed_estimator.p_speed_greater(lane-1, -1)
        
        if cars[lane-1, (y+2)%self.width]:
            return self.speed_estimator.p_speed_greater(lane-1, -2)
        
        return 0.0


    # assess the risk of being tailgated and crash (getting too close)
    def tail_danger_level(self, cars, lane, y, speed):

        # not possible to be tail-gated
        if speed < -2:
            return 0

        if speed == -2:
            if cars[lane, (y+1)%self.width]:
                return 0.33
            else:
                # not possible to be tail-gated if there is no car directly behind
                return 0

        # got car tail gating
        if cars[lane, (y+1)%self.width]:
            return 0.67

        return 0

    def check_deadlock_triangle(self, cars, lane, y):

        if lane == 0:
            return False
        # no problem if the lane has higher than 1 speed
        if self.speed_estimator.max_speed[lane] != -1:
            return False

        px = lane
        py = y
        assert cars[px, py] == 0

        # find the first car in front
        while(cars[px, py] == 0 and py > 0):
            py -= 1

        x_start = px
        y_start = py

        

        while(px > 0):
            if cars[px, py + 1]:
                py += 1
            # we reached the first position that has no car on this lane
            elif cars[px - 1, py + 1] and self.speed_estimator.max_speed[px - 1] == -1:
                # there is car blocking on top, contintue
                px -= 1
                py += 1
            else:
                break
        
        if px == 0:
            py = self.width - 1

        x_end = px
        y_end = py

        assert x_start >= x_end
        assert y_start <= y_end

        # print(x_start, y_start, x_end, y_end)

        # we are in the deadzone if y is at least y_end
        return y <= y_end

    '''
    An example agent that just output a random action.
    '''
    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.
        
        For example, you might want to load the appropriate neural networks weight 
        in this method.
        '''
        test_case_id = kwargs.get('test_case_id')
        
        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        # fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        # gamma               = kwargs.get('gamma')
        self.width =  kwargs.get('width')
        if self.width == None:
            self.width = 50
        self.lanes = kwargs.get('lanes')
        if self.lanes == None:
            self.lanes = 10

        # since agent_speed_range is given, we can fix the actions
        self.UP = 0
        self.DOWN = 1
        # this is forward[-1] position
        self.FORWARD = 2 + (agent_speed_range[1] - agent_speed_range[0] + 1)
        self.agent_max_speed = agent_speed_range[0]
        self.agent_min_speed = agent_speed_range[1] 

        self.cars = 0
        self.agent = 1
        self.finish_position = 2
        self.occupancy_trails = 3

        self.speed_estimator = SpeedEstimator(self.lanes, self.width)
        

        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''

    def reset(self, state, *args, **kwargs):
        ''' 
        [OPTIONAL]
        Reset function of the agent which is used to reset the agent internal state to prepare for a new environement.
        As its name suggests, it will be called after every `env.reset`.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * None
        '''
        '''
        # Uncomment to help debugging
        print('>>> RESET >>>')
        print('state:', state)
        '''
        self.speed_estimator = SpeedEstimator(self.lanes, self.width)

    def step(self, state, *args, **kwargs):

        ''' 
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        
        # Uncomment to help debugging
        # print('>>> STEP >>>')
        # print('state:', state.shape)
        
        self.speed_estimator.update(state[self.cars], state[self.occupancy_trails])
        # self.speed_estimator.print()

        

        agent_lane, agent_y = self.get_position(state, self.agent)

        # run 3 times first, straight
        if agent_lane == self.lanes - 1 and agent_y >= self.width - 3:
            return int(self.FORWARD - 1)

        # print(agent_lane, agent_y)

        # calculate maximum alloable forward speed, without crashing for sure (but may be crashed from behind though)
        max_speed = self.find_max_speed_no_crash(state[self.cars], agent_lane, agent_y)

        # assume the goal is always on the top row, then we are set to just go forward
        if agent_lane == 0:
            return int(self.FORWARD + max_speed)


        # decide if we should move up
        up_danger = self.up_lane_danger_level(state[self.cars], agent_lane, agent_y)

        # print(up_danger)
        if up_danger < 0.1:
            # we want to go up here, however, we want to avoid the triangle deadlock, if there are concecusive speed 1 lanes

            if not self.check_deadlock_triangle(state[self.cars], agent_lane - 1, agent_y):
                return int(self.UP)

        # if the upper lane is always slow, we should move as fast as possible
        if self.speed_estimator.max_speed[agent_lane-1] == -1:
            return int(self.FORWARD + max_speed)

        return int(self.FORWARD -1)

    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')
        
        # Uncomment to help debugging
        # print('>>> UPDATE >>>')
        # print('state:', state)
        # print('action:', action)
        # print('reward:', reward)
        # print('next_state:', next_state)
        # print('done:', done)
        # print('info:', info)
        