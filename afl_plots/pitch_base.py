import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc
from matplotlib import rcParams
import numpy as np

from afl_plots.dimensions import create_pitch_dims

class BasePitch(ABC):
    """ A class for plotting AFL pitches in Matplotlib

    Parameters
    ----------
        
    """
    
    def __init__(self, 
                 half=False,
                 pitch_colour=None,
                 line_colour=None,
                 line_alpha=1,
                 line_width=2,
                 linestyle=None,
                 line_zorder=0.9,
                 pad_left=20,
                 pad_right=20,
                 pad_bottom=20,
                 pad_top=20,
                 pitch_length=None,
                 pitch_width=None,
                 axis=False,
                 label=False,
                 tick=False):
        super().__init__()
        
        self.half = half
        self.pitch_colour = pitch_colour
        if self.pitch_colour is None:
            self.pitch_colour = rcParams['axes.facecolor']
        self.line_colour=line_colour
        if self.line_colour is None:
            self.line_colour = rcParams['grid.color']
        self.line_alpha=line_alpha
        self.line_width=line_width
        self.linestyle=linestyle
        self.line_zorder=line_zorder
        self.pad_left=pad_left
        self.pad_right=pad_right
        self.pad_bottom=pad_bottom
        self.pad_top=pad_top
        self.pitch_length=pitch_length
        self.pitch_width=pitch_width
        self.axis=axis
        self.label=label
        self.tick=tick
               
        # vertical
        self.vertical = None
        
        # Grid
        self.ax_aspect = None
        # Plotting
        self.kde_clip = None
        self.hexbin_gridsize = None
        self.hex_extent = None
        
        # data checks
        # self._validation_checks()
        
        # Pitch Dimensions
        self.dim = create_pitch_dims(self.pitch_width, self.pitch_length)
        
        # Goal Posts
        self.goal_left = np.array([[self.dim.left, self.dim.goal_bottom],
                                   [self.dim.left, self.dim.goal_top]])
        self.goal_right = np.array([[self.dim.right, self.dim.goal_bottom],
                                   [self.dim.right, self.dim.goal_top]])    
        
        # Behind Posts  
        self.behind_left = np.array([[self.dim.left, self.dim.behind_bottom],
                                   [self.dim.left, self.dim.behind_top]])
        self.behind_right = np.array([[self.dim.right, self.dim.behind_bottom],
                                   [self.dim.right, self.dim.behind_top]])     
        
        # # Padding
        # for pad in ['pad_left', 'pad_right', 'pad_bottom', 'pad_top']:
        #     if getattr(self, pad) is None:
        #         setattr(self, pad, 4)
        
        # Set extent
        self._set_extent()
        
        # validate padding
        # self._validate_pad()
        
        # Line Properties
        self.line_properties = {'linewidth':self.line_width, 'alpha':self.line_alpha,
                                'color':self.line_colour, 'zorder':self.line_zorder,
                                'linestyle':self.linestyle}
        
        # Rectangle Properties
        self.rect_properties = {'fill': False, 'linewidth':self.line_width, 'alpha':self.line_alpha,
                                'color':self.line_colour, 'zorder':self.line_zorder,
                                'linestyle':self.linestyle}
        
        # Arc Properties
        self.arc_properties = {'linewidth':self.line_width, 'alpha':self.line_alpha,
                               'color':self.line_colour, 'zorder':self.line_zorder,
                               'linestyle':self.linestyle}
        # Ellipse Properties
        self.ellipse_properties = {'fill': False, 'linewidth':self.line_width, 'alpha':self.line_alpha,
                                   'color':self.line_colour, 'zorder':self.line_zorder,
                                   'linestyle':self.linestyle}

        
    # def _validation_checks(self):
        
    #     # pitch validation
        
    #     # type checks
        
    #     # axis/label warnings
    #     return
        
    # def _validate_pad(self):
    #     # make sure padding not too large for pitch
    #     return
        
    
    
    def draw(self, ax=None, figsize=None, nrows=1, ncols=1, tight_layout=True, constrained_layout=False):
        """ Draw the specified AFL pitch(es). Can draw on existing axes.

        Args:
            ax (_type_, optional): _description_. Defaults to None.
            figsize (_type_, optional): _description_. Defaults to None.
            nrows (int, optional): _description_. Defaults to 1.
            ncols (int, optional): _description_. Defaults to 1.
            tight_layout (bool, optional): _description_. Defaults to True.
            constrained_layout (bool, optional): _description_. Defaults to False.
        """
        
        if figsize is None:
            figsize = rcParams['figure.figsize']
        if ax is None:
            fig, axs = self._setup_subplots(nrows, ncols, figsize, constrained_layout)
            fig.set_tight_layout(tight_layout)
            for axis in axs.flat:
                self._draw_ax(axis)
            if axs.size == 1:
                axs = axs.item()
            return fig, axs
        
        self._draw_ax(ax)
        return None
    
    @staticmethod
    def _setup_subplots(nrows, ncols, figsize, constrained_layout):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                constrained_layout=constrained_layout)
        if (nrows == 1) & (ncols == 1):
            axs = np.array([axs])
            
        return fig, axs
    
    def _draw_ax(self, ax):
        self._set_axes(ax)
        self._draw_pitch_markings(ax)
        self._draw_goals(ax)
        
    def _set_axes(self, ax):
        # Set axis on/off, labels, grid, ticks
        self.set_visible(ax, spine_bottom=self.axis, spine_top=self.axis, spine_left=self.axis, spine_right=self.axis,
                         grid=False, tick=self.tick, label=self.label)
        
        # Set limits and aspect
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_aspect(self.dim.aspect)
               
    def _draw_pitch_markings(self, ax):
        
        self._draw_boundary(ax)
        self._draw_centre(ax)
        self._draw_inside_50(ax)
        
    def _draw_boundary(self, ax):
        # Top Pitch Boundary
        self._draw_arc(ax, x=0, y=self.dim.behind_top, 
                       width=self.pitch_length, height = self.dim.boundary_width,
                       theta1=0, theta2=180, **self.arc_properties)
        # Bottom Pitch Boundary
        self._draw_arc(ax, x=0, y=self.dim.behind_bottom, 
                       width=self.pitch_length, height = self.dim.boundary_width,
                       theta1=180, theta2=360, **self.arc_properties)
        
    def _draw_inside_50(self, ax):
        # Left Inside 50
        self._draw_arc(ax, 
                       x=self.dim.left, y=0, 
                       width=self.dim.inside_50_radius*2, height = self.dim.inside_50_radius*2,
                       theta1=360-self.dim.inside_50_angle, theta2=self.dim.inside_50_angle, **self.arc_properties)        
        # Right Inside 50
        self._draw_arc(ax, 
                x=self.dim.right, y=0, 
                width=self.dim.inside_50_radius*2, height = self.dim.inside_50_radius*2,
                theta1=180-self.dim.inside_50_angle, theta2=180+self.dim.inside_50_angle, **self.arc_properties)  
        
    def _draw_centre(self, ax):
        # Centre Square
        self._draw_rectangle(ax, 
                             x=-self.dim.centre_square/2, y=-self.dim.centre_square/2, 
                             width=self.dim.centre_square, height=self.dim.centre_square,
                             **self.rect_properties)
        # Centre Circle Inner
        self._draw_ellipse(ax, 
                           x=0, y=0, 
                           width = self.dim.centre_circle_inner, height = self.dim.centre_circle_inner,
                           **self.ellipse_properties)
        # Centre Circle Outer
        self._draw_ellipse(ax, 
                           x=0, y=0, 
                           width = self.dim.centre_circle_outer, height = self.dim.centre_circle_outer,
                           **self.ellipse_properties)        
    def _draw_goals(self, ax):
        # Left Goal
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left], 
                        y=[self.dim.goal_bottom, self.dim.goal_top],
                        **self.line_properties)
        # Right Goal
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right], 
                        y=[self.dim.goal_bottom, self.dim.goal_top],
                        **self.line_properties)
        # Left Goal Posts
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left_goal_post], 
                        y=[self.dim.goal_top, self.dim.goal_top],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left_goal_post], 
                        y=[self.dim.goal_bottom, self.dim.goal_bottom],
                        **self.line_properties)
        # Right Goal Posts
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right_goal_post], 
                        y=[self.dim.goal_top, self.dim.goal_top],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right_goal_post], 
                        y=[self.dim.goal_bottom, self.dim.goal_bottom],
                        **self.line_properties)
        # Left Behind
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left], 
                        y=[self.dim.goal_bottom, self.dim.behind_bottom],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left], 
                        y=[self.dim.goal_top, self.dim.behind_top],
                        **self.line_properties)
        # Right Behind
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right], 
                        y=[self.dim.goal_bottom, self.dim.behind_bottom],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right], 
                        y=[self.dim.goal_top, self.dim.behind_top],
                        **self.line_properties)
        # Left Behind Posts
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left_behind_post], 
                        y=[self.dim.behind_top, self.dim.behind_top],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left_behind_post], 
                        y=[self.dim.behind_bottom, self.dim.behind_bottom],
                        **self.line_properties)
        # Right Behind Posts
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right_behind_post], 
                        y=[self.dim.behind_top, self.dim.behind_top],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right_behind_post], 
                        y=[self.dim.behind_bottom, self.dim.behind_bottom],
                        **self.line_properties)
        # Left Goal Square
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left_goal_square], 
                        y=[self.dim.goal_top, self.dim.goal_top],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.left, self.dim.left_goal_square], 
                        y=[self.dim.goal_bottom, self.dim.goal_bottom],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.left_goal_square, self.dim.left_goal_square], 
                        y=[self.dim.goal_bottom, self.dim.goal_top],
                        **self.line_properties)
        # Right Goal Square
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right_goal_square], 
                        y=[self.dim.goal_top, self.dim.goal_top],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.right, self.dim.right_goal_square], 
                        y=[self.dim.goal_bottom, self.dim.goal_bottom],
                        **self.line_properties)
        self._draw_line(ax, 
                        x=[self.dim.right_goal_square, self.dim.right_goal_square], 
                        y=[self.dim.goal_bottom, self.dim.goal_top],
                        **self.line_properties)
        
    def grid(self, figheight=9, nrows=1, ncols=2, grid_height=0.715, grid_width=0.95, space = 0.05,
             left =None, bottom = None, endnote_height = 0.065, endnote_space = 0.01,
             title_height = 0.15, title_space=0.01, axis=True):
        """Helper to create a grid of pitches in a specified location.

        Args:
            figheight (int, optional): _description_. Defaults to 9.
            nrows (int, optional): _description_. Defaults to 1.
            ncols (int, optional): _description_. Defaults to 2.
            grid_height (float, optional): _description_. Defaults to 0.715.
            grid_width (float, optional): _description_. Defaults to 0.95.
            space (float, optional): _description_. Defaults to 0.05.
            left (_type_, optional): _description_. Defaults to None.
            bottom (_type_, optional): _description_. Defaults to None.
            endnote_height (float, optional): _description_. Defaults to 0.065.
            endnote_space (float, optional): _description_. Defaults to 0.01.
            title_height (float, optional): _description_. Defaults to 0.15.
            title_space (float, optional): _description_. Defaults to 0.01.
            axis (bool, optional): _description_. Defaults to True.
            
        Returns:
            fig (matplotlib.figure.Figure): figure
            axs (dict[label, Axs]): A dictionary mapping the laels to the Axes objects - ['pitch', 'title', 'endnote]
        """
        
        dim = self._grid_dimensions(ax_aspect=self.ax_aspect, figheight=figheight, nrows=nrows, ncols=ncols, 
                                    grid_height=grid_height, grid_width=grid_width, space=space, left=left, bottom=bottom,
                                    endnote_height=endnote_height, endnote_space=endnote_space, title_height=title_height, 
                                    title_space=title_space)
        
        left_pad = (np.abs(self.visible_pitch - self.extent)[0] / np.abs(self.extent[1] - self.extent[0])) * dim['axwidth']
        right_pad = (np.abs(self.visible_pitch - self.extent)[1] / np.abs(self.extent[1] - self.extent[0])) * dim['axwidth']
        
        fig, axs = self._draw_grid(dimensions=dim, left_pad = left_pad, right_pad = right_pad, axis=axis, grid_key='pitch')
        
        if endnote_height > 0 or title_height > 0:
            for ax in np.asarray(axs['pitch']).flat:
                self.draw(ax=ax)
        else:
            for ax in np.asarray(axs).flat:
                self.draw(ax=ax)
        
        return fig, axs
    
    @staticmethod
    def _grid_dimensions(self, ax_aspect=1, figheight=9, nrows=1, ncols=1, grid_height=0.715, grid_width=0.95, space=0.05,
                         left=None, bottom=None, endnote_height=0, endnote_space=0.01, title_height=0, title_space=0.01):
        """ A helper to calculate the grid dimensions.

        Args:
            ax_aspect (int, optional): _description_. Defaults to 1.
            figheight (int, optional): _description_. Defaults to 9.
            nrows (int, optional): _description_. Defaults to 1.
            ncols (int, optional): _description_. Defaults to 1.
            grid_height (float, optional): _description_. Defaults to 0.715.
            grid_width (float, optional): _description_. Defaults to 0.95.
            space (float, optional): _description_. Defaults to 0.05.
            left (_type_, optional): _description_. Defaults to None.
            bottom (_type_, optional): _description_. Defaults to None.
            endnote_height (int, optional): _description_. Defaults to 0.
            endnote_space (float, optional): _description_. Defaults to 0.01.
            title_height (int, optional): _description_. Defaults to 0.
            title_space (float, optional): _description_. Defaults to 0.01.

        Returns:
            dimensions (dict[dimension, value]): A dictionary holding the axes and figure dimensions.
        """
        
        # dictionary for holding dimensions
        dimensions = {'figheight': figheight, 'nrows': nrows, 'ncols': ncols,
                    'grid_height': grid_height, 'grid_width': grid_width,
                    'title_height': title_height, 'endnote_height': endnote_height,
                    }

        if left is None:
            left = (1 - grid_width) / 2

        if title_height == 0:
            title_space = 0

        if endnote_height == 0:
            endnote_space = 0

        error_msg_height = ('The axes extends past the figure height. '
                            'Reduce one of the bottom, endnote_height, endnote_space, grid_height, '
                            'title_space or title_height so the total is ≤ 1.')
        error_msg_width = ('The grid axes extends past the figure width. '
                           'Reduce one of the grid_width or left so the total is ≤ 1.')

        axes_height = (endnote_height + endnote_space + grid_height +
                   title_height + title_space)
        if axes_height > 1:
            raise ValueError(error_msg_height)

        if bottom is None:
            bottom = (1 - axes_height) / 2

        if bottom + axes_height > 1:
            raise ValueError(error_msg_height)

        if left + grid_width > 1:
            raise ValueError(error_msg_width)

        dimensions['left'] = left
        dimensions['bottom'] = bottom
        dimensions['title_space'] = title_space
        dimensions['endnote_space'] = endnote_space
        
        if (nrows > 1) and (ncols > 1):
            dimensions['figwidth'] = figheight * grid_height / grid_width * (((1 - space) * ax_aspect *
                                                                            ncols / nrows) +
                                                                            (space * (ncols - 1) / (
                                                                                    nrows - 1)))
            dimensions['spaceheight'] = grid_height * space / (nrows - 1)
            dimensions['spacewidth'] = dimensions['spaceheight'] * figheight / dimensions['figwidth']
            dimensions['axheight'] = grid_height * (1 - space) / nrows

        elif (nrows > 1) and (ncols == 1):
            dimensions['figwidth'] = figheight * grid_height / grid_width * (
                    1 - space) * ax_aspect / nrows
            dimensions['spaceheight'] = grid_height * space / (nrows - 1)
            dimensions['spacewidth'] = 0
            dimensions['axheight'] = grid_height * (1 - space) / nrows

        elif (nrows == 1) and (ncols > 1):
            dimensions['figwidth'] = figheight * grid_height / grid_width * (space + ax_aspect * ncols)
            dimensions['spaceheight'] = 0
            dimensions['spacewidth'] = grid_height * space * figheight / dimensions['figwidth'] / (
                    ncols - 1)
            dimensions['axheight'] = grid_height

        else:  # nrows=1, ncols=1
            dimensions['figwidth'] = figheight * grid_height * ax_aspect / grid_width
            dimensions['spaceheight'] = 0
            dimensions['spacewidth'] = 0
            dimensions['axheight'] = grid_height

        dimensions['axwidth'] = dimensions['axheight'] * ax_aspect * figheight / dimensions['figwidth']

        return dimensions

    def _draw_grid(self, dimensions, left_pad=0, right_pad=0, axis=True, grid_key='grid'):
        """ A helper to create a grid of axes in a specified location

        Parameters
        ----------
        dimensions : dict[dimension, value]
            A dictionary holding the axes and figure dimensions.
            This is created via the _grid_dimensions function.
        left_pad, right_pad : float, default 0
            The padding for the title and endnote. Usually the endnote and title
            are flush to the sides of the axes grid. With the padding option you can
            indent the title and endnote so that there is a gap between the grid axes
            and the title/endnote. The padding units are fractions of the figure width.
        axis : bool, default True
            Whether the endnote and title axes are 'on'.
        grid_key : str, default grid
            The dictionary key for the main axes in the grid.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axs : dict[label, Axes]
            A dictionary mapping the labels to the Axes objects.
        """
        dims = dimensions
        bottom_coordinates = np.tile(dims['spaceheight'] + dims['axheight'],
                                    reps=dims['nrows'] - 1).cumsum()
        bottom_coordinates = np.insert(bottom_coordinates, 0, 0.)
        bottom_coordinates = np.repeat(bottom_coordinates, dims['ncols'])
        grid_bottom = dims['bottom'] + dims['endnote_height'] + dims['endnote_space']
        bottom_coordinates = bottom_coordinates + grid_bottom
        bottom_coordinates = bottom_coordinates[::-1]

        left_coordinates = np.tile(dims['spacewidth'] + dims['axwidth'],
                                reps=dims['ncols'] - 1).cumsum()
        left_coordinates = np.insert(left_coordinates, 0, 0.)
        left_coordinates = np.tile(left_coordinates, dims['nrows'])
        left_coordinates = left_coordinates + dims['left']

        fig = plt.figure(figsize=(dims['figwidth'], dims['figheight']))
        axs = []
        for idx, bottom_coord in enumerate(bottom_coordinates):
            axs.append(fig.add_axes((left_coordinates[idx], bottom_coord,
                                    dims['axwidth'], dims['axheight'])))
        axs = np.squeeze(np.array(axs).reshape((dims['nrows'], dims['ncols'])))
        if axs.size == 1:
            axs = axs.item()
        result_axes = {grid_key: axs}

        title_left = dims['left'] + left_pad
        title_width = dims['grid_width'] - left_pad - right_pad

        if dims['title_height'] > 0:
            ax_title = fig.add_axes(
                (title_left, grid_bottom + dims['grid_height'] + dims['title_space'],
                title_width, dims['title_height']))
            if axis is False:
                ax_title.axis('off')
            result_axes['title'] = ax_title

        if dims['endnote_height'] > 0:
            ax_endnote = fig.add_axes((title_left, dims['bottom'],
                                    title_width, dims['endnote_height']))
            if axis is False:
                ax_endnote.axis('off')
            result_axes['endnote'] = ax_endnote

        if dims['title_height'] == 0 and dims['endnote_height'] == 0:
            return fig, result_axes[grid_key]  # no dictionary if just grid
        return fig, result_axes  # else dictionary

    
    # def jointgrid(self, figheight=9, left=None, grid_width=0.95, bottom=None, 
    #               endnote_height=0.065, endnote_space = 0.01, grid_height=0.715, title_space = 0.01,
    #               title_height = 0.15,
    #               space=0, marginal=0.1,
    #               ax_left=True, ax_top = True, ax_right = True, ax_bottom = False,
    #               axis=True):
    #     """ Create a grid with a pitch at the center and (marginal) axes at sides,

    #     Args:
    #         figheight (int, optional): _description_. Defaults to 9.
    #         left (_type_, optional): _description_. Defaults to None.
    #         grid_width (float, optional): _description_. Defaults to 0.95.
    #         bottom (_type_, optional): _description_. Defaults to None.
    #         endnote_height (float, optional): _description_. Defaults to 0.065.
    #         endnote_space (float, optional): _description_. Defaults to 0.01.
    #         grid_height (float, optional): _description_. Defaults to 0.715.
    #         title_space (float, optional): _description_. Defaults to 0.01.
    #         title_height (float, optional): _description_. Defaults to 0.15.
    #         space (int, optional): _description_. Defaults to 0.
    #         marginal (float, optional): _description_. Defaults to 0.1.
    #         ax_left (bool, optional): _description_. Defaults to True.
    #         ax_top (bool, optional): _description_. Defaults to True.
    #         ax_right (bool, optional): _description_. Defaults to True.
    #         ax_bottom (bool, optional): _description_. Defaults to False.
    #         axis (bool, optional): _description_. Defaults to True.
    #     """
        
        
    ## Abstract methods for drawing attributes - defined in Pitch/VerticalPitch)
    
    def _scale_pad(self):
        """Implement a method for scaling"""
        
    def _set_extent(self):
        """ Implement a method to set the pitch extents. """
    
    @abstractmethod
    def _draw_rectangle(self, ax, x, y, width, height, **kwargs):
        """ Implement a method to draw a rectangle on an axes.  """
        
    @abstractmethod
    def _draw_line(self, ax, x, y, **kwargs):
        """ Implement a method to draw a line on an axes. """
    
    @abstractmethod
    def _draw_ellipse(self, ax, x, y, width, height, **kwargs):
        """ Implement a method to draw a ellipses on an axes.  """
        
    @abstractmethod
    def _draw_arc(self, ax, x, y, width, height, theta1, theta2, **kwargs):
        """ Implement a method to draw an arc on an axes.  """
        
    @staticmethod
    @abstractmethod
    def _reverse_if_vertical(x, y):
        """ Implement a method to reverse x and y coordinates if drawing on a vertical pitch """
        
    @staticmethod
    @abstractmethod
    def _reverse_vertices_if_vertical(vert):
        """ Implement a method to reverse vertices if drawing on a vertical pitch """
        
    @staticmethod
    @abstractmethod
    def _reverse_annotate_if_vertical(annotate):
        """ Implement a method to reverse annotations if drawing on a vertical pitch """ 
        
    ### Plotting methods to be defined in pitch_plot.py (BasePitchPlot)
    
    @abstractmethod
    def plot(self, x, y, ax=None, **kwargs):
        """ Implement a wrapper for matplotlib.axes.Axes.plot. """
    
    # @abstractmethod
    # def scatter(self, x, y, rotation_degrees=None, marker=None, ax=None, **kwargs):
    #     """ Implement a wrapper for matplotlib.axes.Axes.scatter. """
    
    # @abstractmethod
    # def _reflect_2d(self, x, y, standardised=False):
    #     """ Implement a method to reflect points in pitch sides. """
        
    # @abstractmethod
    # def kdeplot(self, x, y, ax=None, **kwargs):
    #     """ Implement a wrapper for seaborn.kdeplot. """
        
    # @abstractmethod
    # def hexbin(self, x, y, ax=None, **kwargs):
    #     """ Implement a wrapper for matplotlib.axes.Axes.hexbin. """
        
    # @abstractmethod
    # def polygon(self, x, y, ax=None, **kwargs):
    #     """ Implement a method to add polygons to the pitch. """
        
    # @abstractmethod
    # def goal_angle(self, x, y, ax=None, goal="right", **kwargs):
    #     """ Implement a method to plot a triangle between a point and goal posts / behind posts. """
        
    # @abstractmethod
    # def annotate(self, text, xy, xytext=None, ax=None, **kwargs):
    #     """ Implement a wrapper for matplotlib.axes.Axes.annotate. """
        
    # @abstractmethod
    # def bin_statistic(self, x, y, values=None, statistic="count", bins=(5,4), 
    #                   normalize=False, standardized=False):
    #     """ Calculate 2d binned statistics for arbitrarily shaped bins. """
        
    # @abstractmethod
    # def heatmap(self, x, y, ax=None, **kwargs):
    #     """ Implement drawing heatmaps for arbitrarily shaped bins. """
        
    # @abstractmethod
    # def flow(self, xstart, ystart, xend, yend, bins=(5,4),
    #          arrow_type='same', arrow_length=5, color=None, ax=None, **kwargs):
    #     """ Implement a flow diagram with arrows showing average direction and
    #         a heatmap showing counts in each bin """
        
    # @abstractmethod
    # def arrows(self, xstart, ystart, xend, yend, *args, ax=None, **kwargs):
    #     """ Implement a method to plot arrows. """
    
    # @abstractmethod
    # def lines(self, xstart, ystart, xend, yend, color=None, n_segments=100,
    #           comet=False, transparent = False, alpha_start=0.01, alpha_end=1,
    #           cmap=None, ax=None, **kwargs):
    #     """ Implement method to plot lines. """
    
    # @abstractmethod
    # def convexhull(self, x, y):
    #     """ calculate a Convex Hull from a set of coordinates. """  
        
    # @abstractmethod
    # def voronoi(self, x, y, teams):
    #     """ Calculate the Voronoi polygons for each team. """ 
    
    # @abstractmethod
    # def calculate_angle_and_distance(self, xstart, ystart, xend, yend, standardized=False, degrees=False):
    #     """ Calculate the angle and distance from a start and end location. """
    
    
    ## Setting axes visibility
    @staticmethod
    def set_visible(ax, spine_bottom=False, spine_top=False, spine_left=False, spine_right=False,
                    grid=False, tick=False, label=False):
        """ Helper method to set visibility of matplotlib spines, grid, ticks. Set to invisible by default.

        Args:
            ax (_type_): _description_
            spine_bottom (bool, optional): _description_. Defaults to False.
            spine_top (bool, optional): _description_. Defaults to False.
            spine_left (bool, optional): _description_. Defaults to False.
            spine_right (bool, optional): _description_. Defaults to False.
            grid (bool, optional): _description_. Defaults to False.
            tick (bool, optional): _description_. Defaults to False.
            label (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        ax.spines['bottom'].set_visible(spine_bottom)
        ax.spines['top'].set_visible(spine_top)
        ax.spines['left'].set_visible(spine_left)
        ax.spines['right'].set_visible(spine_right)
        ax.grid(grid)
        ax.tick_params(bottom=tick, top=tick, left=tick, right=tick,
                       labelbottom=label, labeltop=label, labelleft=label, labelright=label)
    
    ## Used for getting angles for inside 50 arcs.
    @staticmethod   
    def create_ellipse_arc(center_x, center_y, semi_major_axis, semi_minor_axis, rotation_angle, start_angle, end_angle):
        # Generate theta values from start_angle to end_angle
        theta = np.linspace(start_angle, end_angle, 1000)

        # Compute the x and y coordinates of the arc
        x = center_x + semi_major_axis * np.cos(theta) * np.cos(rotation_angle) - semi_minor_axis * np.sin(theta) * np.sin(rotation_angle)
        y = center_y + semi_major_axis * np.cos(theta) * np.sin(rotation_angle) + semi_minor_axis * np.sin(theta) * np.cos(rotation_angle)

        return x, y

    @staticmethod
    def inside_fifty_intersesct(goal_x, goal_y, boundary_x, boundary_y):
    
        distance_to_50 = 100
        for x, y in zip(boundary_x, boundary_y):
            x_diff = goal_x - x
            y_diff = goal_y - y
            
            new_distance_to_50 = abs((x_diff**2 + y_diff**2)**0.5 - 50)

            if new_distance_to_50 < distance_to_50:
                distance_to_50 = new_distance_to_50
                intersect_x, intersect_y = x, y
        
        return intersect_x, intersect_y
    
    @staticmethod
    def get_fifty_angle(goal_x, goal_y, intersect_x, intersect_y):
    
        return np.arctan((intersect_y - goal_y) / (intersect_x - goal_x))
        
    