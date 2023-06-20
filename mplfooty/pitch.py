import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

from mplfooty.pitch_plot import BasePitchPlot

class Pitch(BasePitchPlot):
    
    def _scale_pad(self):
        self.pad_left = self.pad_left * self.dim.aspect
        self.pad_right = self.pad_right * self.dim.aspect
        
    def _set_extent(self):
        extent = np.array([self.dim.left, self.dim.right, self.dim.bottom, self.dim.top],
                          dtype=np.float32)
        pad = np.array([-self.pad_left, self.pad_right, -self.pad_bottom, self.pad_top],
                       dtype=np.float32)
        visible_pad = np.clip(np.array([self.pad_left, self.pad_right, self.pad_bottom, self.pad_top], dtype=np.float32),
                              a_min=None, a_max=0.)
        visible_pad[[0, 2]] = - visible_pad[[0, 2]]
        
        if self.half:
            extent[0] = 0
            visible_pad[0] = - self.pad_left
        if self.dim.invert_y:
            pad[2:] = -pad[2:]
            visible_pad[2:] = - visible_pad[2:]
        self.extent = extent + pad
        self.ax_aspect = (abs(self.extent[1] - self.extent[0]) / (abs(self.extent[3] - self.extent[2]) * self.dim.aspect))
        self.visible_pitch = extent + visible_pad
        
        if self.half:
            extent[0] = extent[0] - min(self.pad_left, self.dim.pitch_length/2)

        # hexbin
        self.hexbin_gridsize = (17, 8)
        self.hex_extent = np.array([self.dim.left, self.dim.right,
                                    min(self.dim.bottom, self.dim.top),
                                    max(self.dim.bottom, self.dim.top)], dtype = np.float32)
        # kdeplot
        
        # lines
        
        # vertical for lines/arrows
        self.vertical = False

        
    def _draw_rectangle(self, ax, x, y, width, height, **kwargs):
        if self.dim.invert_y:
            height = - height
        rectangle = patches.Rectangle((x, y), width, height, **kwargs)
        ax.add_patch(rectangle)
    
    def _draw_line(self, ax, x, y, **kwargs):
        line = Line2D(x, y, **kwargs)
        ax.add_artist(line)
        
    def _draw_ellipse(self, ax, x, y, width, height, **kwargs):
        ellipse = patches.Ellipse((x, y), width, height, **kwargs)
        ax.add_patch(ellipse)
        
    def _draw_arc(self, ax, x, y, width, height, theta1, theta2, **kwargs):
        arc = patches.Arc((x, y), width, height, theta1=theta1, theta2=theta2, **kwargs)
        ax.add_patch(arc)
        
    @staticmethod
    def _reverse_if_vertical(x, y):
        return x, y
    
    @staticmethod
    def _reverse_vertices_if_vertical(vert):
        return vert
    
    @staticmethod
    def _reverse_annotate_if_vertical(annotate):
        return annotate