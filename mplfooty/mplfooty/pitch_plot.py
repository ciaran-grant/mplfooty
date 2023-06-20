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

from mplfooty.pitch_base import BasePitch

_BinnedStatisticResult = namedtuple('BinnedStatisticalResult',
                                    ('statistic', 'x_grid', 'y_grid', 'cx', 'cy', 'binnumber', 'inside'))

class BasePitchPlot(BasePitch):
    
    def plot(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.plot,
        which automatically flips the x and y coordinates if the pitch is vertical.
        
        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.plot.
        
        Returns
        -------
        lines : A list of Line2D objects representing the plotted data.
        
        Examples
        --------
        >>> from mplfooty import Pitch
        >>> pitch = Pitch(pitch_width=135, pitch_length=165)
        >>> fig, ax = pitch.draw()
        >>> pitch.plot([30, 35, 20], [30, 19, 40], ax=ax)
        """
        
        self.validate_ax(ax)
        x, y = self._reverse_if_vertical(x, y)
        return ax.plot(x, y, **kwargs)
        
    
    def scatter(self, x, y, marker=None, ax=None, **kwargs):
        """ Utility wrapper around matplotlib.axes.Axes.scatter,
        which automatically flips the x and y coordinates if the pitch is vertical.
        You can optionally use a football marker with marker='football' and rotate markers with
        rotation_degrees.
        
        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        rotation_degrees: array-like or scalar, default None.
            Rotates the marker in degrees, clockwise. 0 degrees is facing the direction of play.
            In a horizontal pitch, 0 degrees is this way →, in a vertical pitch,
            0 degrees is this way ↑
        marker: MarkerStyle, optional
            The marker style. marker can be either an instance of the class or the
            text shorthand for a particular marker. Defaults to None, in which case it takes
            the value of rcParams["scatter.marker"] (default: 'o') = 'o'.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.scatter.
        
        Returns
        -------
        paths : matplotlib.collections.PathCollection
                
        Examples
        --------
        >>> from mplfooty import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.scatter(30, 30, ax=ax)
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
        
    def kdeplot(self, x, y, ax=None, **kwargs):
        """ Utility wrapper around seaborn.kdeplot,
        which automatically flips the x and y coordinates
        if the pitch is vertical and clips to the pitch boundaries.
        
        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to seaborn.kdeplot.
        
        Returns
        -------
        contour : matplotlib.contour.ContourSet
        
        Examples
        --------
        >>> from mplfooty import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(pitch_width=135, pitch_length=165, line_zorder=2)
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=-100/2, high=100/2, size=36)
        >>> y = np.random.uniform(low=-90/2, high=90/2, size=36)
        >>> pitch.kdeplot(x, y, ax=ax, thresh=0, fill = True, color = "red", levels=100)
        """
        
        self.validate_ax(ax)
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size.")
        
        x, y = self._reverse_if_vertical(x, y)
                        
        kde = sns.kdeplot(x=x, y=y, ax=ax, **kwargs)
        
        # Clip to pitch boundary
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
        """ Utility wrapper around matplotlib.axes.Axes.hexbin,
        which automatically flips the x and y coordinates if the pitch is vertical and
        clips to the pitch boundaries.
        
        Parameters
        ----------
        x, y : array-like or scalar.
            Commonly, these parameters are 1D arrays.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        mincnt : int > 0, default: 1
            If not None, only display cells with more than mincnt number of points in the cell.
        gridsize : int or (int, int), default: (17, 8) for Pitch/ (17, 17) for VerticalPitch
            If a single int, the number of hexagons in the x-direction. The number of hexagons
            in the y-direction is chosen such that the hexagons are approximately regular.
            Alternatively, if a tuple (nx, ny), the number of hexagons in the x-direction
            and the y-direction.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.hexbin.
        
        Returns
        -------
        polycollection : matplotlib.collections.PolyCollection
        
        Examples
        --------
        >>> from mplfooty import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(pitch_width=135, pitch_length=165, line_zorder=2, line_colour="#000009")
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=-82.5, high=100.5, size=10000)
        >>> y= np.random.uniform(low=-82.5, high=67.5, size=10000)
        >>> pitch.hexbin(x, y, edgecolors = "black", ax=ax, cmap="Reds", gridsize=(10, 5))
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

        # Clip to pitch boundary
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
        """ Plot polygons.
        Automatically flips the x and y vertices if the pitch is vertical.
        
        Parameters
        ----------
        verts: verts is a sequence of (verts0, verts1, ...)
            where verts_i is a numpy array of shape (number of vertices, 2).
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to
            matplotlib.patches.Polygon
        
        Returns
        -------
        list of matplotlib.patches.Polygon
        
        Examples
        --------
        >>> from mplfooty import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(pitch_width=135, pitch_length=165, label=True, axis=True)
        >>> fig, ax = pitch.draw()
        >>> shape1 = np.array([[-50, 2], [80, 30], [40, -30], [-40, 20]])
        >>> shape2 = np.array([[70, -70], [-60, 50], [40, 40]])
        >>> verts = [shape1, shape2]
        >>> pitch.polygon(verts, color='red', alpha=0.3, ax=ax)
        """
        
        self.validate_ax(ax)
        patch_list = []
        for vert in verts:
            vert = np.asarray(vert)
            vert = self._reverse_vertices_if_vertical(vert)
            polygon = Polygon(vert, closed=True, **kwargs)
            patch_list.append(polygon)
            ax.add_patch(polygon)
            
        # Clip to pitch boundary 
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
        See: https://matplotlib.org/stable/api/collections_api.html.
        Valid Collection keyword arguments: edgecolors, facecolors, linewidths, antialiaseds,
        transOffset, norm, cmap
        
        Parameters
        ----------
        x, y: array-like or scalar.
            Commonly, these parameters are 1D arrays. These should be the coordinates on the pitch.
        goal: str default 'right'.
            The goal to plot, either 'left' or 'right'.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to
             matplotlib.collections.PathCollection.
        
        Returns
        -------
        Polygon : matplotlib.patches.Polygon
        
        Examples
        --------
        >>> from mplfooty import Pitch
        >>> pitch = Pitch(pitch_width=135, pitch_length=165)
        >>> fig, ax = pitch.draw()
        >>> pitch.goal_angle(50, 30, color = "red", alpha=0.3, ax=ax)
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
        which automatically flips the xy and xytext coordinates if the pitch is vertical.
        Annotate the point xy with text.
        See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html
        
        Parameters
        ----------
        text : str
            The text of the annotation.
        xy : (float, float)
            The point (x, y) to annotate.
        xytext : (float, float), optional
            The position (x, y) to place the text at. If None, defaults to xy.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.annotate.
        
        Returns
        -------
        annotation : matplotlib.text.Annotation
        
        Examples
        --------
        >>> from mplfooty import Pitch
        >>> pitch = Pitch(pitch_width=135, pitch_length=165)
        >>> fig, ax = pitch.draw()
        >>> pitch.annotate(text = "Test", xytext = (-30,0), xy=(40,40), ha='center', va='center', ax=ax, 
                           arrowprops=dict(facecolor="black"))
        """
        
        self.validate_ax(ax)
        xy = self._reverse_annotate_if_vertical(xy)
        if xytext is not None:
            xytext = self._reverse_annotate_if_vertical(xytext)
            
        return ax.annotate(text, xy, xytext, **kwargs)
        
                
    def bin_statistic(self, x, y, values=None, statistic="count", bins=(5,4), 
                      normalize=False):
        """ Calculates binned statistics using scipy.stats.binned_statistic_2d.

        This method automatically sets the range, changes the scipy defaults,
        and outputs the grids and centers for plotting.

        The default statistic has been changed to count instead of mean.
        The default bins have been set to (5,4).

        Parameters
        ----------
        x, y, values : array-like or scalar.
            Commonly, these parameters are 1D arrays.
            If the statistic is 'count' then values are ignored.
        dim : mplfooty pitch dimensions
            Automatically populated when using Pitch/ VerticalPitch class
        statistic : string or callable, optional
            The statistic to compute (default is 'count').
            The following statistics are available: 'count' (default),
            'mean', 'std', 'median', 'sum', 'min', 'max', 'circmean' or a user-defined function. See:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
        bins : int or [int, int] or array_like or [array, array], optional
            The bin specification.
            * the number of bins for the two dimensions (nx = ny = bins),
            * the number of bins in each dimension (nx, ny = bins),
            * the bin edges for the two dimensions (x_edge = y_edge = bins),
            * the bin edges in each dimension (x_edge, y_edge = bins).
                If the bin edges are specified, the number of bins will be,
                (nx = len(x_edge)-1, ny = len(y_edge)-1).
        normalize : bool, default False
            Whether to normalize the statistic by dividing by the total.

        Returns
        -------
        bin_statistic : dict.
            The keys are 'statistic' (the calculated statistic),
            'x_grid' and 'y_grid (the bin's edges), cx and cy (the bin centers)
            and 'binnumber' (the bin indices each point belongs to).
            'binnumber' is a (2, N) array that represents the bin in which the observation falls
            if the observations falls outside the pitch the value is -1 for the dimension. The
            binnumber are zero indexed and start from the top and left handside of the pitch.

        Examples
        --------
        >>> from mplfooty import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(pitch_width=135, pitch_length=165, line_zorder=2, pitch_colour='black')
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=-165/2, high=165/2, size=1000)
        >>> y= np.random.uniform(low=-135/2, high=135/2, size=1000)
        >>> stats = pitch.bin_statistic(x, y, bins=(10, 8))
        >>> pitch.heatmap(stats, edgecolors="black", cmap="hot", ax=ax)
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
        """ Utility wrapper around matplotlib.axes.Axes.pcolormesh
        which automatically flips the x_grid and y_grid coordinates if the pitch is vertical.

        See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html

        Parameters
        ----------
        stats : dict.
            This should be calculated via bin_statistic().
            The keys are 'statistic' (the calculated statistic),
            'x_grid' and 'y_grid (the bin's edges), and cx and cy (the bin centers).
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        vertical : bool, default False
            If the orientation is vertical (True), then the code switches the x and y coordinates.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.pcolormesh.

        Returns
        -------
        mesh : matplotlib.collections.QuadMesh

        Examples
        --------
        >>> from mplfooty import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(pitch_width=135, pitch_length=165, line_zorder=2, pitch_colour='black')
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=-165/2, high=165/2, size=1000)
        >>> y= np.random.uniform(low=-135/2, high=135/2, size=1000)
        >>> stats = pitch.bin_statistic(x, y, bins=(10, 8))
        >>> pitch.heatmap(stats, edgecolors="black", cmap="hot", ax=ax)
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
        """ Labels the heatmap(s) and automatically flips the coordinates if the pitch is vertical.
        
        Parameters
        ----------
        stats : A dictionary or list of dictionaries.
            This should be calculated via bin_statistic_positional() or bin_statistic().
        str_format : str
            A format string passed to str_format.format() to format the labels.
        exclude_zeros : bool
            Whether to exclude zeros when labelling the heatmap.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.annotate.
        
        Returns
        -------
        annotations : A list of matplotlib.text.Annotation.
        
        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> import matplotlib.patheffects as path_effects
        >>> pitch = Pitch(pitch_width=135, pitch_length=165, line_zorder=2, pitch_colour='black')
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=-165/2, high=165/2, size=1000)
        >>> y= np.random.uniform(low=-135/2, high=135/2, size=1000)
        >>> stats = pitch.bin_statistic(x, y, bins=(10, 8))
        >>> pitch.heatmap(stats, edgecolors="black", cmap="hot", ax=ax)
        >>> stats['statistic'] = stats['statistic'].astype(int)
        >>> path_eff = [path_effects.Stroke(linewidth=0.5, foreground='#22312b')]
        >>> text = pitch.label_heatmap(stats, color='white', ax=ax, fontsize=10, ha='center', va='center')
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
        Quiver uses locations and direction vectors usually.
        Here these are instead calculated automatically
        from the start and endpoints of the arrow.
        The function also automatically flips the x and y coordinates if the pitch is vertical.

        Plot a 2D field of arrows.
        See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html

        Parameters
        ----------
        xstart, ystart, xend, yend: array-like or scalar.
            Commonly, these parameters are 1D arrays.
            These should be the start and end coordinates of the lines.
        C: 1D or 2D array-like, optional
            Numeric data that defines the arrow colors by colormapping via norm and cmap.
            This does not support explicit colors.
            If you want to set colors directly, use color instead.
            The size of C must match the number of arrow locations.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        vertical : bool, default False
            If the orientation is vertical (True), then the code switches the x and y coordinates.
        width : float, default 4
            Arrow shaft width in points.
        headwidth : float, default 3
            Head width as a multiple of the arrow shaft width.
        headlength : float, default 5
            Head length as a multiple of the arrow shaft width.
        headaxislength : float, default: 4.5
            Head length at the shaft intersection.
            If this is equal to the headlength then the arrow will be a triangular shape.
            If greater than the headlength then the arrow will be wedge shaped.
            If less than the headlength the arrow will be swept back.
        color : color or color sequence, optional
            Explicit color(s) for the arrows. If C has been set, color has no effect.
        linewidth or linewidths or lw : float or sequence of floats
            Edgewidth of arrow.
        edgecolor or ec or edgecolors : color or sequence of colors or 'face'
        alpha : float or None
            Transparency of arrows.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.quiver.

        Returns
        -------
        PolyCollection : matplotlib.quiver.Quiver

        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch()
        >>> fig, ax = pitch.draw()
        >>> pitch.arrows(20, 20, 45, 80, ax=ax)
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
        """ Calculates the angle in radians counter-clockwise and the distance
        between a start and end location. Where the angle 0 is this way →
        (the straight line from left to right) in a horizontally orientated pitch
        and this way ↑ in a vertically orientated pitch.
        The angle goes from 0 to 2pi. To convert the angle to degrees clockwise use degrees=True.
        
        Parameters
        ----------
        xstart, ystart, xend, yend: array-like or scalar.
            Commonly, these parameters are 1D arrays.
            These should be the start and end coordinates to calculate the angle between.
        standardized : bool, default False
            Whether the x, y values have been standardized to the 'uefa'
            pitch coordinates (105m x 68m)
        degrees : bool, default False
            If False, the angle is returned in radians counter-clockwise in the range [0, 2pi]
            If True, the angle is returned in degrees clockwise in the range [0, 360].
        
        Returns
        -------
        angle: ndarray
            The default is an array of angles in radians counter-clockwise in the range [0, 2pi].
            Where 0 is the straight line left to right in a horizontally orientated pitch
            and the straight line bottom to top in a vertically orientated pitch.
            If degrees = True, then the angle is returned in degrees clockwise in the range [0, 360]
        distance: ndarray
            Array of distances.
        
        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> pitch = Pitch(pitch_width=135, pitch_length=165)
        >>> pitch.calculate_angle_and_distance(0, 40, 30, 20, degrees=True)
        (array([326.30993247]), array([36.05551275]))
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
        """ Create a flow map by binning the data into cells and calculating the average
        angles and distances.
        
        Parameters
        ----------
        xstart, ystart, xend, yend: array-like or scalar.
            Commonly, these parameters are 1D arrays.
            These should be the start and end coordinates to calculate the angle between.
        bins : int or [int, int] or array_like or [array, array], optional
            The bin specification for binning the data to calculate the angles/ distances.
              * the number of bins for the two dimensions (nx = ny = bins),
              * the number of bins in each dimension (nx, ny = bins),
              * the bin edges for the two dimensions (x_edge = y_edge = bins),
              * the bin edges in each dimension (x_edge, y_edge = bins).
                If the bin edges are specified, the number of bins will be,
                (nx = len(x_edge)-1, ny = len(y_edge)-1).
        arrow_type : str, default 'same'
            The supported arrow types are: 'same', 'scale', and 'average'.
            'same' makes the arrows the same size (arrow_length).
            'scale' scales the arrow length by the average distance
            in the cell (up to a max of arrow_length).
            'average' makes the arrow size the average distance in the cell.
        arrow_length : float, default 5
            The arrow_length for the flow map. If the arrow_type='same',
            all the arrows will be arrow_length. If the arrow_type='scale',
            the arrows will be scaled by the average distance.
            If the arrow_type='average', the arrows_length is ignored
            This is automatically multipled by 100 if using a 'tracab' pitch
            (i.e. the default is 500).
        color : A matplotlib color, defaults to None.
            Defaults to None. In that case the marker color is
            determined by the cmap (default 'viridis').
            and the counts of the starting positions in each bin.
        ax : matplotlib.axes.Axes, default None
            The axis to plot on.
        **kwargs : All other keyword arguments are passed on to matplotlib.axes.Axes.quiver.
        
        Returns
        -------
        PolyCollection : matplotlib.quiver.Quiver
        
        Examples
        --------
        >>> from mplfooty.pitch import Pitch
        >>> pitch = Pitch(pitch_width=135, pitch_length=165, line_zorder=2)
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=-165/2, high=165/2, size=1000)
        >>> y= np.random.uniform(low=-135/2, high=135/2, size=1000)
        >>> xend = x + np.random.uniform(low=-10, high = 10, size = 1000)
        >>> yend = y + np.random.uniform(low=-10, high = 10, size = 1000)
        
        >>> bs_heatmap = pitch.bin_statistic(x, y, statistic='count', bins = (6, 4))
        >>> heatmap = pitch.heatmap(bs_heatmap, ax=ax, cmap="Blues")
        >>> flowmap = pitch.flow(x, y, xend, yend, color = "black", arrow_type='same',
                     arrow_length=10, bins=(6,4), headwidth=2, headlength=2, headaxislength=2,
                     ax=ax)
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
        """ Get lines of Convex Hull for a set of coordinates.
        
        Parameters
        ----------
        x, y: array-like or scalar.
            Commonly, these parameters are 1D arrays. These should be the coordinates on the pitch.
        
        Returns
        -------
        hull_vertices: a numpy array of vertoces [1, num_vertices, [x, y]] of the Convex Hull.
        
        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch()
        >>> pitch = Pitch(pitch_width=135, pitch_length=165)
        >>> x = np.random.uniform(low=-165/2, high=165/2, size=18)
        >>> y= np.random.uniform(low=-135/2, high=135/2, size=18)
        >>> hull = pitch.convexhull(x, y)
        >>> poly = pitch.polygon(hull, ax=ax, facecolor='cornflowerblue', alpha=0.3)
        """
        
        points = np.vstack([x, y]).T
        hull = ConvexHull(points)
        
        return points[hull.vertices].reshape(1, -1, 2)        
        
    def _reflect_2d(self, x, y):
        """ Reflect data in the pitch lines."""
        x = np.ravel(x)
        y = np.ravel(y)
        
        x_limits, y_limits = [self.dim.left, self.dim.right], [self.dim.bottom, self.dim.top]
        
        reflected_data_x = np.r_[x, 2 * x_limits[0] - x, 2 * x_limits[1] - x, x, x]
        reflected_data_y = np.r_[y, y, y, 2 * y_limits[0] - y, 2 * y_limits[1] - y]
        return reflected_data_x, reflected_data_y
        
    def voronoi(self, x, y, teams):
        """ Get Voronoi vertices for a set of coordinates.
        Uses a trick by Dan Nichol (@D4N__ on Twitter) where points are reflected in the pitch lines
        before calculating the Voronoi. This means that the Voronoi extends to
        the edges of the pitch. See:
        https://github.com/ProformAnalytics/tutorial_nbs/blob/master/notebooks/Voronoi%20Reflection%20Trick.ipynb
        Players outside the pitch dimensions are assumed to be standing on the pitch edge.
        This means that their coordinates are clipped to the pitch edges
        before calculating the Voronoi.
        
        Parameters
        ----------
        x, y: array-like or scalar.
            Commonly, these parameters are 1D arrays. These should be the coordinates on the pitch.
        teams: array-like or scalar.
            This splits the results into the Voronoi vertices for each team.
            This can either have integer (1/0) values or boolean (True/False) values.
            team1 is where team==1 or team==True
            team2 is where team==0 or team==False
        
        Returns
        -------
        team1 : a 1d numpy array (length number of players in team 1) of 2d arrays
            Where the individual 2d arrays are coordinates of the Voronoi vertices.
        team2 : a 1d numpy array (length number of players in team 2) of 2d arrays
            Where the individual 2d arrays are coordinates of the Voronoi vertices.
        
        Examples
        --------
        >>> from mplsoccer import Pitch
        >>> import numpy as np
        >>> pitch = Pitch(pitch_width=135, pitch_length=165)
        >>> fig, ax = pitch.draw()
        >>> x = np.random.uniform(low=-100/2, high=100/2, size=36)
        >>> y = np.random.uniform(low=-90/2, high=90/2, size=36)
        >>> teams = np.array([0] * 18 + [1] * 18)
        >>> pitch.scatter(x[teams == 0], y[teams == 0], color='red', ax=ax)
        >>> pitch.scatter(x[teams == 1], y[teams == 1], color='blue', ax=ax)
        >>> team1, team2 = pitch.voronoi(x, y, teams)
        >>> team1_poly = pitch.polygon(team1, ax=ax, color='red', alpha=0.3)
        >>> team2_poly = pitch.polygon(team2, ax=ax, color='blue', alpha=0.3)
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
    
        
    