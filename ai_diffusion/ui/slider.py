from __future__ import annotations

from PyQt5.QtWidgets import QSlider


class Slider(QSlider):
    """QSlider whose click beside the handle always moves 10% of the range.

    That click is a "page" step in Qt, and every slider here used to set the page
    step by hand (1, 2, 10, 16, 50 ...), so the same gesture jumped a different
    amount on every slider - 50% of the range on the control preset slider, 8% on
    the resolution slider. One fraction for all of them instead.

    The range is often set in pieces (setMinimum/setMaximum), and C++ does not
    route those through setRange, so each entry point recomputes the page step.
    """

    page_fraction = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_page_step()

    def setMinimum(self, minimum: int):
        super().setMinimum(minimum)
        self._update_page_step()

    def setMaximum(self, maximum: int):
        super().setMaximum(maximum)
        self._update_page_step()

    def setRange(self, minimum: int, maximum: int):
        super().setRange(minimum, maximum)
        self._update_page_step()

    def _update_page_step(self):
        span = self.maximum() - self.minimum()
        self.setPageStep(max(1, round(span * self.page_fraction)))
