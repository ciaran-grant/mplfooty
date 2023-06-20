import warnings
from collections import namedtuple
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc, Rectangle, PathPatch, Polygon
from matplotlib.path import Path
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from scipy.stats import binned_statistic_2d, circmean
from scipy.spatial import ConvexHull, Voronoi

from afl_plots.pitch_base import BasePitch

_BinnedStatisticResult = namedtuple('BinnedStatisticalResult',
                                    ('statistic', 'x_grid', 'y_grid', 'cx', 'cy', 'binnumber', 'inside'))

class BasePitchPlot(BasePitch):
    
    def plot(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.plot.
            Automatically flips x, y coordinates if pitch is vertical.

        Args:
            x (np.array) : x axis values
            y (np.array) : y axis values
            ax (matplotlib.axes.Axes, optional) : The axis to plot on. Defaults to None.
            **kwargs : All other keyword arguments are passed to matplotlib.axes.Axes.plot.

        Returns:
            ax: _description_
        """
        
        self.validate_ax(ax)
        x, y = self._reverse_if_vertical(x, y)
        return ax.plot(x, y, **kwargs)
        
    
    def scatter(self, x, y, marker=None, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.scatter.

        Args:
            x (np.array) : x axis values
            y (np.array) : y axis values
            marker (MarkerStyle, optional): Marker style. Defaults to None.
            ax (matplotlib.axes.Axes, optional) : The axis to plot on. Defaults to None.
            **kwargs : All other keyword arguments are passed to matplotlib.axes.Axes.plot.
            
        Returns:
            paths (matplotlib.collections.PathCollection)
        """
        
        self.validate_ax(ax)
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size.")
        
        x, y = self._reverse_if_vertical(x, y)
        
        if marker is None:
            marker = rcParams['scatter.marker']
        
        return ax.scatter(x, y, marker=marker, **kwargs)
    
    # def _reflect_2d(self, x, y, standardised=False):
    #     """ Implement a method to reflect points in pitch sides. """
        
    def kdeplot(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around seaborn.kdeplot, automatically flips x and y coordinates if vertical.

        Args:
            x (np.array) : x axis values
            y (np.array) : y axis values
            ax (matplotlib.axes.Axes, optional) : The axis to plot on. Defaults to None.
            **kwargs : All other keyword arguments are passed to matplotlib.axes.Axes.plot.
        Returns:
            contour (matplotlib.contour.ContourSet)
        """
        
        self.validate_ax(ax)
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size.")
        
        x, y = self._reverse_if_vertical(x, y)
                        
        kde = sns.kdeplot(x=x, y=y, ax=ax, **kwargs)
        
        top_boundary_arc = Arc((0, self.dim.behind_top), 
                                        width = self.dim.pitch_length, 
                                        height = self.dim.boundary_width, 
                                        theta1=0, theta2=180
                                        )

        bottom_boundary_arc = Arc((0, self.dim.behind_bottom), 
                                    width = self.dim.pitch_length, 
                                    height = self.dim.boundary_width, 
                                    theta1=180, theta2=360
                                    )

        pitch_boundary_vertices = np.concatenate([top_boundary_arc.get_verts()[:-1], bottom_boundary_arc.get_verts()[:-1]])

        A = Polygon(pitch_boundary_vertices, color= "w", zorder=-1)
        ax.add_patch(A)
        for col in ax.collections:
            col.set_clip_path(A)
                
        return kde
        
        
    def hexbin(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.hexbin.

        Args:
            x (_type_): _description_
            y (_type_): _description_
            ax (_type_, optional): _description_. Defaults to None.
        """
        
        
        self.validate_ax(ax)
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size.")
        
        mask = np.isnan(x) | np.isnan(y)
        x = x[~mask]
        y = y[~mask]
        
        x, y = self._reverse_if_vertical(x, y)
        
        mincnt = kwargs.pop('mincnt', 1)
        gridsize = kwargs.pop('gridsize', self.hexbin_gridsize)
        extent = kwargs.pop('extent', self.hex_extent)
        
        hexbin = ax.hexbin(x, y, mincnt = mincnt, gridsize=gridsize, extent=extent, **kwargs)

        top_boundary_arc = Arc((0, self.dim.behind_top), 
                                        width = self.dim.pitch_length, 
                                        height = self.dim.boundary_width, 
                                        theta1=0, theta2=180
                                        )

        bottom_boundary_arc = Arc((0, self.dim.behind_bottom), 
                                    width = self.dim.pitch_length, 
                                    height = self.dim.boundary_width, 
                                    theta1=180, theta2=360
                                    )

        pitch_boundary_vertices = np.concatenate([top_boundary_arc.get_verts()[:-1], bottom_boundary_arc.get_verts()[:-1]])

        A = Polygon(pitch_boundary_vertices, color= "w", zorder=-1)
        ax.add_patch(A)
        for col in ax.collections:
            col.set_clip_path(A)

        return hexbin        
        
    def polygon(self, verts, ax=None, **kwargs):
        """ Plots polygons on the pitch.

        Args:
            verts (_type_): _description_
            ax (_type_, optional): _description_. Defaults to None.
        
        Returns:
        
        Examples:
        
        """
        
        self.validate_ax(ax)
        patch_list = []
        for vert in verts:
            vert = np.asarray(vert)
            vert = self._reverse_vertices_if_vertical(vert)
            polygon = Polygon(vert, closed=True, **kwargs)
            patch_list.append(polygon)
            ax.add_patch(polygon)
         
        top_boundary_arc = Arc((0, self.dim.behind_top), 
                                width = self.dim.pitch_length, 
                                height = self.dim.boundary_width, 
                                theta1=0, theta2=180
                                )

        bottom_boundary_arc = Arc((0, self.dim.behind_bottom), 
                                    width = self.dim.pitch_length, 
                                    height = self.dim.boundary_width, 
                                    theta1=180, theta2=360
                                    )

        pitch_boundary_vertices = np.concatenate([top_boundary_arc.get_verts()[:-1], bottom_boundary_arc.get_verts()[:-1]])

        A = Polygon(pitch_boundary_vertices, color= "w", zorder=-1)
        ax.add_patch(A)
        for patch in patch_list:
            patch.set_clip_path(A)    
             
        return patch_list
        
             
    def goal_angle(self, x, y, ax=None, goal="right", **kwargs):
        """ Plot a polygon with the angle to the goal using matplotlib.patches.Polygon.

        Args:
            x (_type_): _description_
            y (_type_): _description_
            ax (_type_, optional): _description_. Defaults to None.
            goal (str, optional): _description_. Defaults to "right".
            
        Returns:
        
        Examples:
        """
        
        self.validate_ax(ax)
        valid_goal = ['left', 'right']
        if goal not in valid_goal:
            raise TypeError(f'Invalid argument: goal should be in {valid_goal}')
        
        x = np.ravel(x)
        y = np.ravel(y)
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size.")
        
        goal_coordinates = self.goal_right if goal == 'right' else self.goal_left
        verts = np.zeros((x.size, 3, 2))
        verts[:, 0, 0] = x
        verts[:, 0, 1] = y
        verts[:, 1:, :] = np.expand_dims(goal_coordinates, 0)
        return self.polygon(verts, ax=ax, **kwargs)


    def annotate(self, text, xy, xytext=None, ax=None, **kwargs):
        """ Utility wrapper around ax.annotate

        Args:
            text (_type_): _description_
            xy (_type_): _description_
            xytext (_type_, optional): _description_. Defaults to None.
            ax (_type_, optional): _description_. Defaults to None.
        
        Returns:
        
        Examples:
        
        """
        
        self.validate_ax(ax)
        xy = self._reverse_annotate_if_vertical(xy)
        if xytext is not None:
            xytext = self._reverse_annotate_if_vertical(xytext)
            
        return ax.annotate(text, xy, xytext, **kwargs)
        
                
    def bin_statistic(self, x, y, values=None, statistic="count", bins=(5,4), 
                      normalize=False, standardized=False):
        """ Calculates binned statistics using scipy.stats.binned_statistics_2d

        Args:
            x (_type_): _description_
            y (_type_): _description_
            values (_type_, optional): _description_. Defaults to None.
            statistic (str, optional): _description_. Defaults to "count".
            bins (tuple, optional): _description_. Defaults to (5,4).
            normalize (bool, optional): _description_. Defaults to False.
            standardized (bool, optional): _description_. Defaults to False.
            
        Returns:
            bin_statistic (dict): Keys: 'statistic' - calculated statistic
                                        'x_grid', 'y_grid' - bin's edges
                                        'cx', 'cy' - bin's centers
                                        'binnumber' - bin indices each point belongs to. (2, N) array starting from 0 
                                                                                          and top left of pitch.

        Examples:
        
        """
        
        x = np.ravel(x)
        y = np.ravel(y)
                   
        if x.size != y.size:
            raise ValueError("x and y must be the same size.")
        
        if statistic == "mean":
            statistic == np.nanmean
        elif statistic == "std":
            statistic == np.nanstd
        elif statistic == "median":
            statistic == np.nanmedian
        elif statistic == "sum":
            statistic == np.nansum        
        elif statistic == "min":
            statistic == np.nanmin
        elif statistic == "max":
            statistic == np.nanmax
        if (values is None) & (statistic == 'count'):
            values = x
        if (values is None) & (statistic != "count"):
            raise ValueError("values on which to calculate the statistic are missing.")
        if self.dim.invert_y:
            pitch_range = [[self.dim.left, self.dim.right], [self.dim.top, self.dim.bottom]]
            y = self.dim.bottom - y
        else:
            pitch_range = [[self.dim.left, self.dim.right], [self.dim.bottom, self.dim.top]]
        
        statistic, x_edge, y_edge, binnumber = binned_statistic_2d(x, y, values, statistic=statistic,
                                                                   bins=bins, range=pitch_range,
                                                                   expand_binnumbers=True)
                    
        statistic = np.flip(statistic.T, axis=0)
        if statistic.ndim == 3:
            num_y, num_x, _ = statistic.shape
        else:
            num_y, num_x = statistic.shape
            
        if normalize:
            statistic = statistic / statistic.sum()
            
        binnumber[1, :] = num_y - binnumber[1, :] + 1
        x_grid, y_grid = np.meshgrid(x_edge, y_edge)
        cx, cy = np.meshgrid(x_edge[:-1] + 0.5 * np.diff(x_edge), y_edge[:-1] + 0.5 * np.diff(y_edge))
        
        if not self.dim.invert_y:
            y_grid = np.flip(y_grid, axis=0)
            cy = np.flip(cy, axis=0)
        
        ## if outside the pitch, set bin number to minus 1
        mask_x_out = np.logical_or(binnumber[0, :] == 0,
                                   binnumber[0, :] == num_x + 1)
        binnumber[0, mask_x_out] = -1
        binnumber[0, ~mask_x_out] = binnumber[0, ~mask_x_out] - 1
        
        mask_y_out = np.logical_or(binnumber[1, :] == 0,
                                   binnumber[1, :] == num_y + 1)
        binnumber[1, mask_y_out] = -1
        binnumber[1, ~mask_y_out] = binnumber[1, ~mask_y_out] - 1
        inside = np.logical_and(~mask_x_out, ~mask_y_out)
        return _BinnedStatisticResult(statistic, x_grid, y_grid, cx, cy, binnumber, inside)._asdict()
        
    def heatmap(self, stats, ax=None, vertical=False, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.pcolormesh.
        

        Args:
            stats (dict): Calculated via binned_statstic.
            ax (_type_, optional): _description_. Defaults to None.
            vertical (bool, optional): _description_. Defaults to False.
            
        Returns:
            mesh (matplotlib.collections.QuadMesh)
            
        Examples:
        
        """
        
        self.validate_ax(ax)
        
        if vertical:
            heatmap = ax.pcolormesh(stats['y_grid'], stats['x_grid'], stats['statistic'], **kwargs)
        
        heatmap = ax.pcolormesh(stats['x_grid'], stats['y_grid'], stats['statistic'], **kwargs)
        
        top_boundary_arc = Arc((0, self.dim.behind_top), 
                                        width = self.dim.pitch_length, 
                                        height = self.dim.boundary_width, 
                                        theta1=0, theta2=180
                                        )

        bottom_boundary_arc = Arc((0, self.dim.behind_bottom), 
                                    width = self.dim.pitch_length, 
                                    height = self.dim.boundary_width, 
                                    theta1=180, theta2=360
                                    )

        pitch_boundary_vertices = np.concatenate([top_boundary_arc.get_verts()[:-1], bottom_boundary_arc.get_verts()[:-1]])

        A = Polygon(pitch_boundary_vertices, color= "w", zorder=-1)
        ax.add_patch(A)
        for col in ax.collections:
            col.set_clip_path(A)
        
        return heatmap
      
    def label_heatmap(self, stats, str_format=None, exclude_zeros=False, ax=None, **kwargs):
        """ Labels the heatmaps.

        Args:
            stats (_type_): _description_
            str_format (_type_, optional): _description_. Defaults to None.
            exclude_zeros (bool, optional): _description_. Defaults to False.
            ax (_type_, optional): _description_. Defaults to None.
            
        Returns:
        
        Examples:
        
        """
        
        self.validate_ax(ax)
        
        if not isinstance(stats, list):
            stats = [stats]
            
        annotation_list = []
        for bin_stat in stats:
            # remove labels outside the plot extents
            mask_x_outside1 = bin_stat['cx'] < self.dim.pitch_extent[0]
            mask_x_outside2 = bin_stat['cx'] > self.dim.pitch_extent[1]
            mask_y_outside1 = bin_stat['cy'] < self.dim.pitch_extent[2]
            mask_y_outside2 = bin_stat['cy'] > self.dim.pitch_extent[3]
            mask_clip = mask_x_outside1 | mask_x_outside2 | mask_y_outside1 | mask_y_outside2
            if exclude_zeros:
                mask_clip = mask_clip | (np.isclose(bin_stat['statistic'], 0.))
            mask_clip = np.ravel(mask_clip)
            
            text =  np.ravel(bin_stat['statistic'])[~mask_clip]
            cx = np.ravel(bin_stat['cx'])[~mask_clip]
            cy = np.ravel(bin_stat['cy'])[~mask_clip]
            for idx, text_str in enumerate(text):
                if str_format is not None:
                    text_str = str_format.format(text_str)
                annotation = self.annotate(text_str, (cx[idx], cy[idx]), ax=ax, **kwargs)
                annotation_list.append(annotation)
        
        return annotation_list
        

        
    def arrows(self, xstart, ystart, xend, yend, *args, ax=None, vertical = False, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.quiver. 

        Args:
            xstart (_type_): _description_
            ystart (_type_): _description_
            xend (_type_): _description_
            yend (_type_): _description_
            ax (_type_, optional): _description_. Defaults to None.
            
        Parameters:
        
        
        Examples:
        
        
        """
        
        self.validate_ax(ax)
        
        units = kwargs.pop('units', 'inches')
        scale_units = kwargs.pop('scale_units', 'xy')
        angles = kwargs.pop('angles', 'xy')
        scale = kwargs.pop('scale', 1)
        width = kwargs.pop('width', 4)
        
        width = width / 72.
        
        xstart = np.ravel(xstart)
        ystart = np.ravel(ystart)
        xend = np.ravel(xend)
        yend = np.ravel(yend)        
        
        if xstart.size != ystart.size:
            raise ValueError("xstart and ystart must be the same size.")
        if xstart.size != xend.size:
            raise ValueError("xstart and xend must be the same size.")
        if ystart.size != yend.size:
            raise ValueError("ystart and yend must be the same size.")
        
        # vectors for direction
        u = xend - xstart
        v = yend - ystart
        
        if vertical:
            ystart, xstart = xstart, ystart
            v, u = u, v
        
        
        q = ax.quiver(xstart, ystart, u, v, *args, units = units,
                      scale_units = scale_units, angles=angles, scale=scale, width=width,
                      **kwargs)
        
        return q
        
    def calculate_angle_and_distance(self, xstart, ystart, xend, yend, degrees = False):
        """ Calculates the angle in radians anti-clockwise and the distance between a start
            and end location.

        Args:
            xstart (_type_): _description_
            ystart (_type_): _description_
            xend (_type_): _description_
            yend (_type_): _description_
            degrees (bool, optional): _description_. Defaults to False.
            
        Returns:
        
        
        Examples:
        
        
        """
        xstart = np.ravel(xstart)
        ystart = np.ravel(ystart)
        xend = np.ravel(xend)
        yend = np.ravel(yend) 

        if xstart.size != ystart.size:
            raise ValueError("xstart and ystart must be the same size.")
        if xstart.size != xend.size:
            raise ValueError("xstart and xend must be the same size.")
        if ystart.size != yend.size:
            raise ValueError("ystart and yend must be the same size.")
        
        x_dist = xend - xstart
        if self.dim.invert_y:
            y_dist = ystart - yend
        else:
            y_dist = yend - ystart
            
        angle = np.arctan2(y_dist, x_dist)
        angle[angle < 0] = 2 * np.pi + angle[angle < 0]    
        
        if degrees:
            angle = np.mod(-np.degrees(angle, 360))
            
        distance = (x_dist ** 2 + y_dist ** 2 ) ** 0.5
        
        return angle, distance
        
    
    def flow(self, xstart, ystart, xend, yend, bins=(5,4),
             arrow_type='same', arrow_length=5, color=None, ax=None, **kwargs):
        """_summary_

        Args:
            xstart (_type_): _description_
            ystart (_type_): _description_
            xend (_type_): _description_
            yend (_type_): _description_
            bins (tuple, optional): _description_. Defaults to (5,4).
            arrow_type (str, optional): _description_. Defaults to 'same'.
            arrow_length (int, optional): _description_. Defaults to 5.
            color (_type_, optional): _description_. Defaults to Noneax=None.
            
        Returns:
        
        
        Examples:
        
        """
        
        
        self.validate_ax(ax)
        
        # binned statistics
        angle, distance = self.calculate_angle_and_distance(xstart, ystart, xend, yend)
        
        bs_distance = self.bin_statistic(xstart, ystart, values=distance, statistic="mean", bins=bins)
        bs_angle = self.bin_statistic(xstart, ystart, values=angle, statistic=circmean, bins=bins)
        
        # arrow length
        if arrow_type == "scale":
            new_d = (bs_distance['statistic'] * arrow_length /
                     np.nan_to_num(bs_distance['statistic']).max(initial=None))
        elif arrow_type == "same":
            new_d = arrow_length
        elif arrow_type == "average":
            new_d = bs_distance['statistic']
        else:
            valid_arrows = ['scale', 'same', 'average']
            raise TypeError(f'Invalid argument: arrow_type should be in {valid_arrows}')
        
        # calculate the end positions of the arrows
        endx = bs_angle['cx'] + (np.cos(bs_angle['statistic']) * new_d)
        if self.dim.invert_y:
            endy = bs_angle['cy'] - (np.sin(bs_angle['statistic']) * new_d)
        else:
            endy = bs_angle['cy'] + (np.sin(bs_angle['statistic']) * new_d)
            
        # get coordinates and convert back if necessary
        cx, cy = bs_angle['cx'], bs_angle['cy']
        
        #plot arrows
        if color is not None:
            return self.arrows(cx, cy, endx, endy, color=color, ax=ax, **kwargs)
        
        bs_count = self.bin_statistic(xstart, ystart, statistic='count', bins=bins)
        return self.arrows(cx, cy, endx, endy, bs_count['statistic'], ax=ax, **kwargs)
            
    
    def convexhull(self, x, y):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        y : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        
        points = np.vstack([x, y]).T
        hull = ConvexHull(points)
        
        return points[hull.vertices].reshape(1, -1, 2)        
        
    def _reflect_2d(self, x, y):
        
        x = np.ravel(x)
        y = np.ravel(y)
        
        x_limits, y_limits = [self.dim.left, self.dim.right], [self.dim.bottom, self.dim.top]
        
        reflected_data_x = np.r_[x, 2 * x_limits[0] - x, 2 * x_limits[1] - x, x, x]
        reflected_data_y = np.r_[y, y, y, 2 * y_limits[0] - y, 2 * y_limits[1] - y]
        return reflected_data_x, reflected_data_y
        
    def voronoi(self, x, y, teams):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        y : _type_
            _description_
        teams : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
             
        x = np.ravel(x)
        y = np.ravel(y)
        teams = np.ravel(teams)
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size.")
        if teams.size != x.size:
            raise ValueError("x and team must be the same size.")
        
        # clip to pitch extent
        extent = self.dim.pitch_extent
        x = x.clip(min=extent[0], max=extent[1])
        y = y.clip(min=extent[2], max=extent[3])
        
        # reflect in pitch lines
        reflect_x, reflect_y = self._reflect_2d(x, y)
        reflect = np.vstack([reflect_x, reflect_y]).T
        
        # create Voronoi
        vor = Voronoi(reflect)
        
        # get region vertices
        regions = vor.point_region[:x.size]
        regions = np.array(vor.regions, dtype='object')[regions]
        region_vertices = []
        for region in regions:
            verts = vor.vertices[region]
            verts[:, 0] = np.clip(verts[:, 0], a_min=extent[0], a_max=extent[1])
            verts[:, 1] = np.clip(verts[:, 1], a_min=extent[2], a_max=extent[3])
            
            region_vertices.append(verts)
            
        region_vertices = np.array(region_vertices, dtype='object')
        
        # separate team1/team2 vertices
        team1 = region_vertices[teams == 0]
        team2 = region_vertices[teams == 1]
        
        return team1, team2
    
    # def lines(self, xstart, ystart, xend, yend, color=None, n_segments=100,
    #           comet=False, transparent = False, alpha_start=0.01, alpha_end=1,
    #           cmap=None, ax=None, **kwargs):
    #     """ Implement method to plot lines. """
    
    @staticmethod
    def validate_ax(ax):
        " Error message if ax is missing. "
        if ax is None:
            msg = "Missing 1 required argument: ax. A Matplotlib axis is required for plotting."
            raise TypeError(msg)
    
        
    