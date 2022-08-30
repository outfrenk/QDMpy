import matplotlib
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider

from pyqdm.app.canvas import GlobalFluorescenceCanvas
from pyqdm.app.widgets.misc import gf_applied_window
from pyqdm.app.widgets.qdm_widget import QDMWidget
import matplotlib.pyplot as plt


class GlobalWidget(QDMWidget):
    def add_odmr(self):
        self.canvas.update_odmr(
            freq=self.qdm.odmr.f_ghz,
            data=self.get_uncorrected_odmr(),
            uncorrected=self.get_current_odmr(),
            corrected=self.get_corrected_odmr(self.global_slider.value()),
        )

    def __init__(self, qdm_instance, *args, **kwargs):
        canvas = GlobalFluorescenceCanvas()
        super().__init__(canvas=canvas, *args, **kwargs)
        self.LOG.debug("GlobalFluorescenceWindow.__init__")
        self.setWindowTitle("Global Fluorescence")

        # layout
        self.add_global_slider()
        self.set_main_window()

        # add plotting elements
        self.add_mean_odmr()
        self.add_light()
        self.add_laser()
        self.add_scalebars()
        self.update_marker()

        # data and uncorrected are swapped so that the markers are always for
        # the uncorrected data. Other widgets will not have this.
        self.canvas.update_odmr(
            freq=self.qdm.odmr.f_ghz,
            data=self.get_uncorrected_odmr(),
            uncorrected=self.get_current_odmr(),
            corrected=self.get_corrected_odmr(self.global_slider.value()),
        )
        # self.canvas.update_odmr_lims()
        self.set_ylim()
        self.update_clims()
        self.canvas.draw_idle()
        plt.tight_layout()

    def add_global_slider(self):
        self.global_label = QLabel(f"Global Fluorescence: {self.qdm.global_factor:.2f}")
        self.global_slider = QSlider()
        self.global_slider.setValue(self.caller.gf_select.value())
        self.global_slider.setRange(0, 100)
        self.global_slider.setOrientation(Qt.Horizontal)
        self.global_slider.valueChanged.connect(self.on_global_slider_change)
        self.applyButton = QPushButton("Apply")
        self.applyButton.clicked.connect(self.apply_global_factor)
        # finish main layout
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.global_label)
        horizontal_layout.addWidget(self.global_slider)
        horizontal_layout.addWidget(self.applyButton)
        self.mainVerticalLayout.addLayout(horizontal_layout)

    def on_global_slider_change(self):
        self.global_label.setText(
            f"Global Fluorescence: {self.global_slider.value() / 100:.2f}"
        )
        self.update_odmr()
        self.canvas.draw_idle()

    def update_odmr(self):
        """
        Update the marker position on the image plots.
        """
        self.canvas.update_odmr(
            freq=self.qdm.odmr.f_ghz,
            data=self.get_uncorrected_odmr(),
            uncorrected=self.get_current_odmr(),
            corrected=self.get_corrected_odmr(self.global_slider.value()),
        )
        # self.canvas.update_odmr_lims()
        self.set_ylim()
    def apply_global_factor(self):
        self.LOG.debug(f"applying global factor {self.global_slider.value() / 100:.2f}")
        self.qdm.odmr.correct_glob_fluorescence(self.global_slider.value() / 100)
        gf_applied_window(self.global_slider.value() / 100)
        self.caller.gf_select.setValue(self.global_slider.value() / 100)
        self.update_odmr()
        self.close()
