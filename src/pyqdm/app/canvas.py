import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyqdm.plotting as qdmplot

POL = ["+", "-"]
FRANGE = ["<", ">"]


class QDMCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    @property
    def data_axes(self):
        return list(self.data.keys())

    @property
    def img_axes(self):
        return (
            list(self.light.keys()) + list(self.laser.keys()) + list(self.data.keys()) + list(self.fluorescence.keys())
        )

    @property
    def odmr_axes(self):
        return list(self.odmr.keys())

    @property
    def has_odmr(self):
        return len(self.odmr_axes) != 0

    @property
    def has_img(self):
        return len(self.img_axes) != 0

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.LOG = logging.getLogger(f"pyQDM.{self.__class__.__name__}")
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.img_dict = {
            "data": None,
            "outlier": None,
            "overlay": None,
            "marker": None,
            "pol": [],
            "frange": [],
            "cax": False,
            "cax_locator": None,
        }
        self.odmr_dict = {
            "data": [[None, None], [None, None]],
            "pol": [],
            "frange": [],
            "mean": [[None, None], [None, None]],
            "fit": [[None, None], [None, None]],
            "corrected": [[None, None], [None, None]],
            "uncorrected": [[None, None], [None, None]],
        }
        self.light = {}  # dict of ax : led img
        self.laser = {}  # laser is a dictionary of dictionaries
        self.fluorescence = {}  # fluorescence is a dictionary of dictionaries
        self.data = {}
        self.odmr = {}  # dictionary of ax : odmr data lines

    def _add_cax(self, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        original_locator = cax.get_axes_locator()
        return cax, original_locator

    def set_img(self):
        for axdict in [self.data, self.laser, self.light, self.fluorescence]:
            for a in axdict:
                a.set(xlabel="px", ylabel="px")
                if axdict == self.data and self.data[a]["cax"] is not None:
                    self.data[a]["cax"].set_ylabel(r"B$_{111}$ [$\mu$T]")
                elif isinstance(axdict[a]["cax"], Axes):
                    axdict[a]["cax"].set_ylabel("intensity [a.u.]")

    def set_odmr(self):
        for a in self.odmr:
            a.set(xlabel="GHz", ylabel="contrast [a.u.]")

    def add_light(self, light, data_dimensions):
        for ax in self.light.keys():
            self.LOG.debug(f"Adding Light image to axis {ax}")
            self.light[ax]["data"] = qdmplot.plot_light_img(
                ax=ax,
                data=light,
                img=self.light[ax]["data"],
                extent=[0, data_dimensions[1], 0, data_dimensions[0]],
            )

    def add_laser(self, laser, data_dimensions):
        for ax in self.laser:
            self.LOG.debug(f"Adding laser to axis {ax}")
            self.laser[ax]["data"] = qdmplot.plot_laser_img(
                ax,
                laser,
                self.laser[ax]["data"],
                data_dimensions=data_dimensions,
            )

    def add_data(self, data, data_dimensions):
        for ax in self.data:
            self.data[ax]["data"] = qdmplot.plot_data(
                ax=ax,
                data=data,
                img=self.data[ax]["data"],
                data_dimensions=data_dimensions,
                aspect="equal",
                origin="lower",
            )

    def add_outlier_masks(self, outlier):
        for axdict in [self.light, self.laser, self.data]:
            for ax in axdict:
                if axdict[ax]["outlier"] is None:
                    self.LOG.debug(f"Adding outlier mask to axis {ax}")
                    axdict[ax]["outlier"] = ax.imshow(
                        outlier,
                        cmap="gist_rainbow",
                        alpha=outlier.astype(float),
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        origin="lower",
                        aspect="equal",
                        zorder=2,
                    )

    def add_scalebars(self, pixelsize):
        for axdict in [self.light, self.laser, self.data]:
            # Create scale bar
            scalebar = ScaleBar(pixelsize, "m", length_fraction=0.25, location="lower left")
            for ax in axdict:
                ax.add_artist(scalebar)

    def add_cax(self, ax, axdict, save=True):
        cax, original_locator = self._add_cax(ax)
        if save:
            axdict[ax]["cax"] = cax
            axdict[ax]["cax_locator"] = original_locator
        else:
            [s.set_visible(False) for s in cax.spines.values()]
            cax.set_xticks([])
            cax.set_yticks([])

    def update_marker(self, x, y):
        for axdict in [self.light, self.laser, self.data, self.fluorescence]:
            for ax in axdict:
                self.LOG.debug(f"Updating marker to axis {ax}")
                axdict[ax]["marker"] = qdmplot.update_marker(
                    ax, x, y, line=axdict[ax]["marker"], marker="X", c="m", mfc="w", zorder=100
                )

    def update_outlier_masks(self, outlier):
        for ax, img in self.outlier.items():
            self.LOG.debug(f"Updating outlier mask to axis {ax}")
            img.set_data(outlier)

    def add_mean_odmr(self, freq, mean):
        for ax in self.odmr:
            for p, f in itertools.product(self.odmr[ax]["pol"], self.odmr[ax]["frange"]):
                self.LOG.debug(f"Adding mean odmr to axis {ax}")

    def update_odmr(self, freq, data=None, fit=None, corrected=None, uncorrected=None, mean=None):
        for ax in self.odmr:
            self.LOG.debug(f"Updating ODMR lines in axis {ax}")
            for p, f in itertools.product(self.odmr[ax]["pol"], self.odmr[ax]["frange"]):
                if data is not None:
                    self.update_data(ax, data, freq, p, f)
                if fit is not None:
                    self.update_fit(ax, freq, fit, p, f)
                if corrected is not None:
                    self.update_corrected(ax, freq, corrected, p, f)
                if uncorrected is not None:
                    self.update_uncorrected(ax, freq, uncorrected, p, f)
                if mean is not None:
                    self.update_mean(ax, freq, mean, p, f)

    def update_mean(self, ax, freq, mean, p, f):
        self.odmr[ax]["mean"][p][f] = qdmplot.update_line(
            ax=ax,
            x=freq[f],
            y=mean[p, f],
            line=self.odmr[ax]["mean"][p][f],
            ls="--",
            zorder=0,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=0.8,
        )

    def update_uncorrected(self, ax, freq, uncorrected, p, f):
        self.odmr[ax]["uncorrected"][p][f] = qdmplot.update_line(
            ax=ax,
            x=freq[f],
            y=uncorrected[p, f],
            line=self.odmr[ax]["uncorrected"][p][f],
            ls="-.",
            zorder=4,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=0.7,
        )

    def update_corrected(self, ax, freq, corrected, p, f):
        self.odmr[ax]["corrected"][p][f] = qdmplot.update_line(
            ax,
            freq[f],
            y=corrected[p, f],
            line=self.odmr[ax]["corrected"][p][f],
            ls="-",
            zorder=3,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=1,
        )

    def update_fit(self, ax, freq, fit, p, f):
        self.odmr[ax]["fit"][p][f] = qdmplot.update_line(
            ax,
            np.linspace(freq[f].min(), freq[f].max(), 200),
            fit[p, f],
            line=self.odmr[ax]["fit"][p][f],
            zorder=2,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=1,
        )

    def update_data(self, ax, data, freq, p, f):
        self.odmr[ax]["data"][p][f] = qdmplot.update_line(
            ax=ax,
            x=freq[f],
            y=data[p, f],
            line=self.odmr[ax]["data"][p][f],
            marker=".",
            ls="",
            zorder=1,
            mfc="w",
        )

    def add_fluorescence(self, fluorescence):
        for ax in self.fluorescence:
            for p, f in itertools.product(self.fluorescence[ax]["pol"], self.fluorescence[ax]["frange"]):
                if p in self.fluorescence[ax]["pol"] and f in self.fluorescence[ax]["frange"]:
                    self.LOG.debug(f"Adding fluorescence of {POL[p]}, {FRANGE[f]} to axis {ax}")
                    self.fluorescence[ax]["data"] = qdmplot.plot_fluorescence(
                        ax, fluorescence[p][f], img=self.fluorescence[ax]["data"]
                    )

    # def update_odmr(self, data=None, fit=None, corrected=None, uncorrected=None):
    #     for ax in self.odmr:
    #         for p, f in itertools.product(self.odmr[ax]["pol"], self.odmr[ax]["frange"]):
    #             if data is not None and self.odmr[ax]["data"][p][f] is not None:
    #                 self.LOG.debug(f"Updating odmr in axis {ax}")
    #                 self.odmr[ax]["data"][p][f].set_ydata(data[p, f])
    #             if fit is not None and self.odmr[ax]["fit"][p][f] is not None:
    #                 self.odmr[ax]["fit"][p][f].set_ydata(fit[p, f])
    #             if corrected is not None and self.odmr[ax]["corrected"][p][f] is not None:
    #                 self.odmr[ax]["corrected"][p][f].set_ydata(corrected[p, f])
    #             if uncorrected is not None and self.odmr[ax]["uncorrected"][p][f] is not None:
    #                 self.odmr[ax]["uncorrected"][p][f].set_ydata(uncorrected[p, f])

    def update_odmr_lims(self):
        self.LOG.debug("updating xy limits for the pixel plots")
        for ax in self.odmr:
            lines = np.array(self.odmr[ax]["data"] + self.odmr[ax]["mean"]).flatten()
            lines = [l for l in lines if l is not None]
            mn = np.nanmin([np.min(l.get_ydata()) for l in lines])
            mx = np.nanmax([np.max(l.get_ydata()) for l in lines])
            ax.set(ylim=(mn * 0.999, mx * 1.001))

    def update_clims(self, use_percentile, percentile):
        self.LOG.debug(f"updating clims for all images with {percentile} percentile ({use_percentile}).")

        for axdict in [self.data, self.laser, self.fluorescence]:
            for ax in axdict:
                if axdict[ax]["data"] is None:
                    continue
                vmin, vmax = qdmplot.get_vmin_vmax(axdict[ax]["data"], percentile, use_percentile)
                norm = qdmplot.get_color_norm(vmin=vmin, vmax=vmax)
                axdict[ax]["data"].set(norm=norm)
                if axdict[ax]["cax"]:
                    qdmplot.update_cbar(
                        img=axdict[ax]["data"],
                        cax=axdict[ax]["cax"],
                        vmin=vmin,
                        vmax=vmax,
                        original_cax_locator=axdict[ax]["cax_locator"],
                    )


class GlobalFluorescenceCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

        self.fig.subplots_adjust(top=0.9, bottom=0.09, left=0.075, right=0.925, hspace=0.28, wspace=0.899)

        spec = self.fig.add_gridspec(ncols=6, nrows=2)

        self.left_mean_odmr_ax = self.fig.add_subplot(spec[0, :3])
        self.right_mean_odmr_ax = self.fig.add_subplot(spec[0, 3:6])

        self.light_ax = self.fig.add_subplot(spec[1, :3])
        self.laser_ax = self.fig.add_subplot(spec[1, 3:])

        self.light_ax.get_shared_x_axes().join(self.light_ax, self.laser_ax)
        self.light_ax.get_shared_y_axes().join(self.light_ax, self.laser_ax)

        # setup the dictionaries for the data
        self.light = {self.light_ax: self.img_dict.copy()}
        self.laser = {self.laser_ax: self.img_dict.copy()}
        cax, original_locator = self._add_cax(self.laser_ax)
        self.laser[self.laser_ax]["cax"] = cax
        self.laser[self.laser_ax]["cax_locator"] = original_locator

        self.odmr = {
            self.left_mean_odmr_ax: self.odmr_dict.copy(),
            self.right_mean_odmr_ax: self.odmr_dict.copy(),
        }
        self.odmr[self.left_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.left_mean_odmr_ax]["frange"] = [0]
        self.odmr[self.right_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.right_mean_odmr_ax]["frange"] = [1]


class FitCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

        spec = self.fig.add_gridspec(ncols=6, nrows=3)
        self.fig.subplots_adjust(top=0.952, bottom=0.076, left=0.06, right=0.959, hspace=0.35, wspace=0.594)
        self.data_ax = self.fig.add_subplot(spec[0:2, :4])
        self.data = {self.data_ax: self.img_dict.copy()}
        self.add_cax(self.data_ax, self.data)

        self.light_ax = self.fig.add_subplot(spec[0, 4:])
        self.light = {self.light_ax: self.img_dict.copy()}
        self.add_cax(self.light_ax, self.light_ax, save=False)

        self.laser_ax = self.fig.add_subplot(spec[1, 4:])
        self.laser = {self.laser_ax: self.img_dict.copy()}
        self.add_cax(self.laser_ax, self.laser)

        self.data_ax.get_shared_x_axes().join(self.data_ax, self.light_ax, self.laser_ax)
        self.data_ax.get_shared_y_axes().join(self.data_ax, self.light_ax, self.laser_ax)

        self.left_odmr_ax = self.fig.add_subplot(spec[2, :3])
        self.right_odmr_ax = self.fig.add_subplot(spec[2, 3:])

        self.odmr = {self.left_odmr_ax: self.odmr_dict.copy(), self.right_odmr_ax: self.odmr_dict.copy()}

        # setup the dictionaries for the data
        self.odmr[self.left_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.left_odmr_ax]["frange"] = [0]
        self.odmr[self.right_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.right_odmr_ax]["frange"] = [1]

        self.set_img()
        self.set_odmr()


class SimpleCanvas(QDMCanvas):
    def __init__(self, dtype, width=5, height=4, dpi=100):
        super().__init__(width=width, height=height, dpi=dpi)
        self.fig.subplots_adjust(top=0.97, bottom=0.082, left=0.106, right=0.979, hspace=0.2, wspace=0.2)

        if "light" in dtype.lower():
            self.light_ax = self.fig.add_subplot(111)
            self.light = {self.light_ax: self.img_dict.copy()}
        elif "laser" in dtype.lower():
            self.laser_ax = self.fig.add_subplot(111)
            cax, original_locator = self._add_cax(self.laser_ax)
            self.laser = {self.laser_ax: self.img_dict.copy()}
            self.laser[self.laser_ax]["cax"] = cax
            self.laser[self.laser_ax]["cax_locator"] = original_locator
        else:
            raise ValueError(f"dtype {dtype} not recognized")


class FittingPropertyCanvasOLD(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.966, bottom=0.06, left=0.056, right=0.985, hspace=0.325, wspace=1.0)

        self.fig = fig

        spec = fig.add_gridspec(ncols=6, nrows=3)

        self.main_ax = fig.add_subplot(spec[0:2, :4])
        divider = make_axes_locatable(self.main_ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)
        self.original_cax_locator = self.cax._axes_locator

        self.led_ax = fig.add_subplot(spec[0, 4:])
        self.led_ax.set_title("reflected light")
        self.led_ax.set_xlabel("px")
        self.led_ax.set_ylabel("px")

        self.laser_ax = fig.add_subplot(spec[1, 4:])
        self.laser_ax.set_title("laser")
        self.laser_ax.set_xlabel("px")
        self.laser_ax.set_ylabel("px")

        self.main_ax.get_shared_x_axes().join(self.main_ax, self.led_ax, self.laser_ax)
        self.main_ax.get_shared_y_axes().join(self.main_ax, self.led_ax, self.laser_ax)

        self.left_ODMR_ax = fig.add_subplot(spec[2, :3])
        self.left_ODMR_ax.set_title("low freq. ODMR")
        self.left_ODMR_ax.set_xlabel("frequency [GHz]")
        self.left_ODMR_ax.set_ylabel("contrast [a.u.]")

        self.right_ODMR_ax = fig.add_subplot(spec[2, 3:])
        self.right_ODMR_ax.set_title("high freq. ODMR")
        self.right_ODMR_ax.set_xlabel("frequency [GHz]")
        self.right_ODMR_ax.set_ylabel("contrast [a.u.]")

        self._is_spectra = [self.left_ODMR_ax, self.right_ODMR_ax]
        self._is_data = [self.main_ax]
        self._is_img = [self.main_ax, self.led_ax, self.laser_ax]
        super().__init__(fig)


class FluoImgCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

        widths = [1, 1]
        heights = [1, 1, 1, 0.1]
        gs = self.fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights)
        self.fig.subplots_adjust(top=0.981, bottom=0.019, left=0.097, right=0.93, hspace=0.47, wspace=0.406)
        self.low_f_mean_odmr_ax = self.fig.add_subplot(gs[0, 0])
        self.high_f_mean_odmr_ax = self.fig.add_subplot(gs[0, 1])

        self.fluo_lowF_pos_ax = self.fig.add_subplot(gs[1, 0])
        self.fluo_highF_pos_ax = self.fig.add_subplot(gs[1, 1])
        self.fluo_lowF_neg_ax = self.fig.add_subplot(gs[2, 0])
        self.fluo_highF_neg_ax = self.fig.add_subplot(gs[2, 1])

        # setup the dictionaries for the fluorescence data
        self.fluorescence = {
            self.fluo_lowF_pos_ax: self.img_dict.copy(),
            self.fluo_highF_pos_ax: self.img_dict.copy(),
            self.fluo_lowF_neg_ax: self.img_dict.copy(),
            self.fluo_highF_neg_ax: self.img_dict.copy(),
        }
        self.fluorescence[self.fluo_lowF_pos_ax]["pol"] = [0]
        self.fluorescence[self.fluo_highF_pos_ax]["pol"] = [0]
        self.fluorescence[self.fluo_lowF_neg_ax]["pol"] = [1]
        self.fluorescence[self.fluo_highF_neg_ax]["pol"] = [1]

        self.fluorescence[self.fluo_lowF_pos_ax]["frange"] = [0]
        self.fluorescence[self.fluo_highF_pos_ax]["frange"] = [1]
        self.fluorescence[self.fluo_lowF_neg_ax]["frange"] = [0]
        self.fluorescence[self.fluo_highF_neg_ax]["frange"] = [1]

        self.add_cax(self.fluo_lowF_pos_ax, self.fluorescence)
        self.add_cax(self.fluo_highF_pos_ax, self.fluorescence)
        self.add_cax(self.fluo_lowF_neg_ax, self.fluorescence)
        self.add_cax(self.fluo_highF_neg_ax, self.fluorescence)

        self.odmr = {
            self.low_f_mean_odmr_ax: self.odmr_dict.copy(),
            self.high_f_mean_odmr_ax: self.odmr_dict.copy(),
        }
        self.odmr[self.low_f_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.low_f_mean_odmr_ax]["frange"] = [0]
        self.odmr[self.high_f_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.high_f_mean_odmr_ax]["frange"] = [1]

        self.set_img()
        self.set_odmr()


class QualityCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=False)
        fig.subplots_adjust(top=0.94, bottom=0.054, left=0.038, right=0.957, hspace=0.15, wspace=0.167)

        self.fig = fig
        widths = [1, 1]
        heights = [1, 1]
        spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)

        self.left_top_ax = fig.add_subplot(spec[0, 0])
        self.right_top_ax = fig.add_subplot(spec[0, 1])
        self.left_bottom_ax = fig.add_subplot(spec[1, 0])
        self.right_bottom_ax = fig.add_subplot(spec[1, 1])

        self.left_top_ax.get_shared_x_axes().join(
            self.left_top_ax, self.left_bottom_ax, self.right_top_ax, self.right_bottom_ax
        )
        self.left_top_ax.get_shared_y_axes().join(
            self.left_top_ax, self.left_bottom_ax, self.right_top_ax, self.right_bottom_ax
        )

        self.ax = np.array([[self.left_top_ax, self.right_top_ax], [self.left_bottom_ax, self.right_bottom_ax]])
        self._is_img = self.ax.flatten()
        self._is_spectra = []

        for a in self.ax.flatten():
            a.set(xlabel="px", ylabel="px")

        self.caxes = np.array([[None, None], [None, None]])
        self.original_cax_locator = np.array([[None, None], [None, None]])

        for p, f in itertools.product(range(2), range(2)):
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(self.ax[p][f])
            self.caxes[p][f] = divider.append_axes("right", size="5%", pad=0.05)
            self.original_cax_locator[p][f] = self.caxes[p][f]._axes_locator

        super().__init__(fig)


if __name__ == "__main__":
    c = GlobalFluorescenceCanvas()
    print("c:", c.data_axes)
