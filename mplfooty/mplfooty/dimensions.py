from dataclasses import dataclass
from typing import Optional

from matplotlib import patches
import numpy as np

@dataclass
class BaseDims:
    """ Base dataclass to hold AFL pitch dimensions. """
    pitch_width: float
    pitch_length: float
    goal_width: float
    behind_width: float
    goal_post_length: float
    behind_post_length: float
    goal_square_length: float
    centre_square: float
    centre_circle_outer: float
    centre_circle_inner: float
    inside_50_radius: float
    
    invert_y: bool
    origin_center: bool
    aspect: Optional[float] = None
    pitch_extent: Optional[np.array] = None

    def setup_dims(self):
        """ Run methods for pitch dimensions. """
        
        self.goals_and_behinds()
        self.boundary_widths()
        self.inside_50_angle()
    
    def boundary_widths(self):
        """ Creates boundary width from pitch and behind posts. """
        
        self.boundary_width = self.pitch_width - self.behind_top*2
            
    def goals_and_behinds(self):
        """ Create goals and behind dimensions.  """
        
        self.left = -self.pitch_length/2
        self.right = - self.left
        self.top = self.pitch_width/2
        self.bottom = - self.top
        
        self.goal_top = self.goal_width/2
        self.goal_bottom = -self.goal_width/2
        
        self.left_goal_square = self.left + self.goal_square_length
        self.right_goal_square = self.right - self.goal_square_length
        
        self.left_goal_post = self.left - self.goal_post_length
        self.right_goal_post = self.right + self.goal_post_length
        
        self.behind_top = self.goal_top + self.behind_width
        self.behind_bottom = self.goal_bottom - self.behind_width
        
        self.left_behind_post = self.left - self.behind_post_length
        self.right_behind_post = self.right + self.behind_post_length
        
        if self.invert_y:
            self.pitch_extent = np.array([self.left, self.right, self.top, self.bottom])
        else:
            self.pitch_extent = np.array([self.left, self.right, self.bottom, self.top])
    
    def inside_50_angle(self):
        """ Calculate angle for inside 50 arc to connect to pitch boundary. """
        
        bottom_boundary_arc = patches.Arc((0, self.behind_bottom), 
                                          width = self.pitch_length, 
                                          height = self.boundary_width, 
                                          theta1=180, theta2=360)
        
        vertices = bottom_boundary_arc.get_verts()
        interpolated_array = self.interpolate_points(vertices, 10)
        distance_to_50 = [abs((((v[0]-self.left)**2 + (v[1]-0)**2)**0.5)-50) for v in interpolated_array]
        closest_vertex_to_50 = interpolated_array[distance_to_50.index(min(distance_to_50))]

        self.inside_50_angle = abs(np.arctan((closest_vertex_to_50[1] - 0) / (closest_vertex_to_50[0] - self.left)) * (180 / np.pi))

    @staticmethod
    def interpolate_points(vertices, num_points):
        interpolated_points = []
        
        for i in range(len(vertices) - 1):
            start_point = vertices[i]
            end_point = vertices[i + 1]
            
            for j in range(num_points):
                t = float(j) / (num_points + 1)
                interpolated_point = (1 - t) * start_point + t * end_point
                interpolated_points.append(interpolated_point)
        
        return np.array(interpolated_points)
        
@dataclass
class VariableCentreDims(BaseDims):
    """ Dataclass holding dimensions for pitches where origin is the centre of the pitch. """
    def __post_init__(self):
        self.setup_dims()

def afl_dims(pitch_width, pitch_length):
    """ Create AFL pitch dimensions. """
    return VariableCentreDims(pitch_width=pitch_width, pitch_length=pitch_length,
                              goal_width=6.4, behind_width=6.4,
                              goal_post_length=15, behind_post_length=10, goal_square_length=9,
                              centre_square=50, centre_circle_outer=10, centre_circle_inner=3,
                              inside_50_radius = 50,
                              aspect=1,
                              invert_y = False, origin_center=True
                              )
    
def create_pitch_dims(pitch_width, pitch_length):
    """ Create pitch dimensions.

    Parameters
    ----------
    pitch_width : float
        Pitch width in metres.
    pitch_length : float
        Pitch length in metres.

    Returns
    -------
    dataclass
        Dataclass holding pitch dimensions.
    """

    return afl_dims(pitch_width, pitch_length)
    
    