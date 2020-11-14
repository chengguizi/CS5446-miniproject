import numpy as np


class SpeedEstimator:
    def __init__(self, lanes=10, width=50):
        self.min_speed = - np.ones(lanes) * 3
        self.max_speed = np.zeros(lanes)
        self.width = width
        self.lanes = lanes

    def update(self, cars, occupancy_trails):
        for x in range(self.lanes):
            for y in range(self.width):
                speed = -1
                # if there is car at the position
                if(cars[x,y] > 0):
                    end_trail = (y+1)%self.width
                    while(occupancy_trails[x,end_trail] > 0 and cars[x, end_trail] == 0):
                        end_trail = (end_trail + 1)%self.width
                        speed -= 1
                
                    # make sure the end of trail is not followed by another car, to exclude ambiguity
                    if(cars[x,end_trail] == 0):
                        # speed estimate could be used
                        if speed < self.max_speed[x]:
                            self.max_speed[x] = speed
                        if speed > self.min_speed[x]:
                            self.min_speed[x] = speed

    def print(self):
        for x in range(self.lanes):
            print(self.min_speed[x], self.max_speed[x])

