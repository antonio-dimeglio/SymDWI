"""Small reusable widgets shared across the sidebar panels."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QButtonGroup,
)
from PySide6.QtCore import Qt, Signal


class Collapsible(QWidget):
    """A disclosure section: a clickable header with a chevron that shows/
    hides a body widget. Used to keep secondary or occasional-use controls
    out of the way until they're needed.

    Signals:
        toggled(bool): Emitted whenever the expanded state changes, with
            the new expanded state.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, body: QWidget, expanded: bool = False, parent=None):
        """Initialize the disclosure section.

        Args:
            title: Text shown next to the chevron in the header.
            body: Widget to show/hide as the section's content. Ownership
                is transferred to this widget's layout.
            expanded: Initial expanded state.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._header = QPushButton()
        self._header.setProperty("variant", "disclosure")
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.clicked.connect(self._on_clicked)
        outer.addWidget(self._header)

        self.body = body
        self.body.setVisible(expanded)
        outer.addWidget(self.body)

        self._title = title
        self._set_header_text(expanded)

    def _set_header_text(self, expanded: bool):
        """Update the header label's chevron glyph to match `expanded`."""
        chevron = "▾" if expanded else "▸"
        self._header.setText(f"{chevron}  {self._title}")

    def _on_clicked(self):
        """Slot for the header button's `clicked` signal; applies the new
        checked state via `set_expanded`."""
        self.set_expanded(self._header.isChecked())

    def set_expanded(self, expanded: bool):
        """Programmatically expand or collapse the section.

        Updates the header's checked state and chevron, shows/hides the
        body widget, and emits :attr:`toggled`.

        Args:
            expanded: True to expand (show the body), False to collapse.

        Returns:
            None.
        """
        self._header.blockSignals(True)
        self._header.setChecked(expanded)
        self._header.blockSignals(False)
        self._set_header_text(expanded)
        self.body.setVisible(expanded)
        self.toggled.emit(expanded)

    def is_expanded(self) -> bool:
        """Return whether the section is currently expanded."""
        return self._header.isChecked()


class SegmentedToggle(QWidget):
    """A compact two-or-more-option toggle, styled as connected buttons.
    Used where a checkbox + hint label would otherwise be needed to convey
    a mode (e.g. "Default" vs "Custom") — the state is legible at a glance
    instead of requiring a tooltip.

    Signals:
        currentChanged(int): Emitted with the index of the newly selected
            option whenever the user clicks a different segment.
    """

    currentChanged = Signal(int)

    def __init__(self, options: list[str], parent=None):
        """Initialize the toggle with a fixed set of mutually-exclusive
        options.

        Args:
            options: Labels for each segment, in display order. The first
                option is selected by default.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []

        for i, label in enumerate(options):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("variant", "segment")
            if i == 0:
                btn.setProperty("segment-pos", "first")
            elif i == len(options) - 1:
                btn.setProperty("segment-pos", "last")
            else:
                btn.setProperty("segment-pos", "mid")
            layout.addWidget(btn)
            self._group.addButton(btn, i)
            self._buttons.append(btn)

        self._buttons[0].setChecked(True)
        self._group.idClicked.connect(self.currentChanged)

    def set_current(self, index: int):
        """Select the option at `index` without emitting a signal.

        Args:
            index: Zero-based index of the option to select. Out-of-range
                values are silently ignored.

        Returns:
            None.
        """
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)

    def current(self) -> int:
        """Return the zero-based index of the currently selected option."""
        return self._group.checkedId()
