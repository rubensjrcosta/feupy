# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Region-of-interest sky map plotting utilities."""

import matplotlib.pyplot as plt
from gammapy.maps import RegionGeom
from regions import CircleSkyRegion, PointSkyRegion

from feupy.visualization.styles.markers.plotting import build_point_kwargs

__all__ = ["ROIMapPlotter"]


class ROIMapPlotter:
    """Plot a sky map containing a region of interest and optional sources.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        Sky coordinate of the ROI center.
    radius : `~astropy.units.Quantity`
        Radius of the ROI in angular units.
    """

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.ax = None

    def customize_legend(self, kwargs_legend=None):
        """Customize and add the legend.

        Parameters
        ----------
        kwargs_legend : dict, optional
            Keyword arguments passed to ``Axes.legend``.
        """
        if kwargs_legend is None:
            kwargs_legend = {
                "bbox_to_anchor": (0, -0.45),
                "ncol": 3,
                "loc": "lower left",
                "markerscale": 0.75,
                "fontsize": 5,
                "labelcolor": "black",
            }

        self.ax.legend(**kwargs_legend)

    def plot_roi(self, color="blue", linestyle="--"):
        """Plot the region of interest.

        Parameters
        ----------
        color : str, optional
            ROI edge color. Default is ``"blue"``.
        linestyle : str, optional
            ROI line style. Default is ``"--"``.

        Returns
        -------
        `matplotlib.axes.Axes`
            Axes containing the ROI.
        """
        region = RegionGeom(CircleSkyRegion(self.center, self.radius))
        self.ax = region.plot_region(
            color=color,
            linestyle=linestyle,
        )

        return self.ax

    def plot_sources(self, sources, ref_markers=None):
        """Plot astronomical sources on the sky map.

        Parameters
        ----------
        sources : object
            Source collection exposing iteration and a ``labels`` attribute.
            Each source must expose a ``position`` attribute.
        ref_markers : dict, optional
            Plotting keyword arguments indexed by source label.
        """
        if ref_markers is None:
            ref_markers = build_point_kwargs(
                sources,
                marker_size=6,
                palette=None,
            )

        for index, source in enumerate(sources):
            kwargs_point = {
                **ref_markers[sources.labels[index]],
                "fillstyle": "full",
                "lw": 0,
            }

            point = RegionGeom(PointSkyRegion(center=source.position))
            self.ax = point.plot_region(
                ax=self.ax,
                facecolor=kwargs_point["color"],
                edgecolor="black",
                kwargs_point=kwargs_point,
            )

    def set_axes(
        self,
        xlabel="R.A. (J2000)",
        ylabel="Dec. (J2000)",
        size=12,
    ):
        """Set axis labels and enable the grid.

        Parameters
        ----------
        xlabel : str, optional
            Label for the x-axis. Default is ``"R.A. (J2000)"``.
        ylabel : str, optional
            Label for the y-axis. Default is ``"Dec. (J2000)"``.
        size : int, optional
            Font size for the axis labels. Default is 12.
        """
        self.ax.set_xlabel(xlabel, size=size)
        self.ax.set_ylabel(ylabel, size=size)
        self.ax.grid(True)

    def add_roi_text(self):
        """Add an annotation describing the ROI radius."""
        self.ax.text(
            0.1,
            0.93,
            f"ROI ({self.radius})",
            transform=self.ax.transAxes,
        )

    def save_plot(self, file_path):
        """Save the current ROI map.

        Parameters
        ----------
        file_path : str or `~pathlib.Path`
            Destination path. If None, the figure is not saved.
        """
        if file_path:
            plt.savefig(
                file_path,
                dpi=300,
                bbox_inches="tight",
            )

    def plot(self, sources=None, file_path=None, **kwargs):
        """Plot the ROI map.

        Parameters
        ----------
        sources : object, optional
            Source collection to overlay on the ROI map.
        file_path : str or `~pathlib.Path`, optional
            Destination path used to save the figure.
        **kwargs : dict
            Optional ``ref_markers`` and ``kwargs_legend`` entries.

        Returns
        -------
        `matplotlib.axes.Axes`
            Axes containing the ROI map.
        """
        self.ax = plt.gca() if self.ax is None else self.ax

        self.plot_roi()

        if sources:
            ref_markers = kwargs.get("ref_markers")
            self.plot_sources(sources, ref_markers)

        kwargs_legend = kwargs.get("kwargs_legend")
        self.customize_legend(kwargs_legend)

        self.set_axes()
        self.add_roi_text()
        self.save_plot(file_path)

        return self.ax
