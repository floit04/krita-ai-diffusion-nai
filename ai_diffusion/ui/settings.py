from __future__ import annotations

from krita import Krita
from PyQt5.QtCore import QMetaObject, QSize, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QCursor,
    QDesktopServices,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QPainter,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyleOption,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, eventloop, util
from ..backend import resources
from ..backend.client import Client, MissingResources, User
from ..backend.cloud_client import CloudClient
from ..backend.nai_client import NaiClient
from ..backend.resources import Arch, ResourceId
from ..backend.server import Server, ServerState
from ..localization import Localization
from ..localization import translate as _
from ..model.connection import ConnectionState, apply_performance_preset
from ..model.properties import Binding
from ..model.root import collect_diagnostics, root
from ..model.updates import UpdateState
from ..settings import ImageFileFormat, PerformancePreset, ServerMode, Settings, settings
from ..style import Style, is_nai_style
from .server import ServerWidget
from .settings_widgets import (
    ComboBoxSetting,
    FileListSetting,
    SettingsTab,
    SliderSetting,
    SpinBoxSetting,
    SwitchSetting,
)
from .style import StylePresets
from .theme import add_header, green, grey, logo, prompt_max_line_count, red, yellow


class InitialSetupWidget(QWidget):
    finished = pyqtSignal(ServerMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 0)
        self.setLayout(layout)

        label_title = QLabel("<b>" + _("Welcome to Image Generation in Krita") + "</b>", self)
        label_sub = QLabel(
            _(
                "To create images, the plugin needs to connect to a backend server. Please choose one of the options below (you can always switch later)."
            ),
            self,
        )
        label_sub.setWordWrap(True)
        layout.addWidget(label_title)
        layout.addWidget(label_sub)
        layout.addSpacing(20)

        def add_option(title: str, desc_text: str, button_text: str, mode: ServerMode):
            header = QLabel("<b>" + title + "</b>", self)
            desc = QLabel(desc_text, self)
            desc.setMaximumWidth(600)
            desc.setWordWrap(True)
            button = QPushButton(button_text, self)
            button.setMinimumHeight(int(1.3 * button.sizeHint().height()))
            button.setMaximumWidth(300)
            button.clicked.connect(self._choose(mode))
            layout.addWidget(header)
            layout.addWidget(desc)
            layout.addWidget(button)
            layout.addSpacing(16)

        add_option(
            _("Option {number}", number=1) + ": " + _("Online Service"),
            _(
                "Generate images via {link}. Create an account to get started. No local installation or powerful hardware needed.",
                link="<a href='https://www.interstice.cloud'>interstice.cloud</a>",
            ),
            _("Login or Sign up"),
            ServerMode.cloud,
        )
        add_option(
            _("Option {number}", number=2) + ": " + _("Local Managed Server"),
            _(
                "Install and run a local ComfyUI server on your machine. Installation and updates are performed automatically by the plugin. Requires a compatible GPU (NVIDIA with at least 6GB VRAM recommended)."
            ),
            _("Start Installation"),
            ServerMode.managed,
        )
        add_option(
            _("Option {number}", number=3) + ": " + _("Custom ComfyUI"),
            _(
                "Connect to an existing installation of ComfyUI. It can be on the same machine, or a remote machine over the network. You are responsible to setup ComfyUI and install required custom nodes and models."
            )
            + "<br><a href='https://docs.interstice.cloud/comfyui-setup'>ComfyUI Setup Guide</a>",
            _("Connect via URL"),
            ServerMode.external,
        )
        layout.addStretch()

    def _choose(self, mode: ServerMode):
        def handler():
            settings.server_mode = mode
            settings.save()
            self.finished.emit(mode)

        return handler


class UserWidget(QFrame):
    _user: User | None = None
    _connections: list[QMetaObject.Connection | Binding]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections = []

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setVisible(False)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self._user_name = QLabel("", self)
        self._user_name.setStyleSheet("font-weight:bold")
        user_name_layout = QHBoxLayout()
        user_name_layout.addWidget(QLabel(_("Account:"), self), 0)
        user_name_layout.addWidget(self._user_name, 1)
        layout.addLayout(user_name_layout)

        self._images_generated = QLabel("", self)
        image_count_layout = QHBoxLayout()
        image_count_layout.addWidget(QLabel(_("Total generated:"), self), 0)
        image_count_layout.addWidget(self._images_generated, 1)
        layout.addLayout(image_count_layout)

        self._tokens_remaining = QLabel("", self)
        self._tokens_remaining.setStyleSheet("font-weight:bold")
        image_remaining_layout = QHBoxLayout()
        image_remaining_layout.addWidget(QLabel(_("Image tokens remaining:"), self), 0)
        image_remaining_layout.addWidget(self._tokens_remaining, 1)
        layout.addLayout(image_remaining_layout)
        layout.addSpacing(8)

        buy_layout = QHBoxLayout()
        layout.addLayout(buy_layout)

        self._buy_tokens5000_button = QPushButton(_("Buy Tokens") + " (5000)", self)
        self._buy_tokens5000_button.clicked.connect(lambda: self._buy_tokens("5000"))
        buy_layout.addWidget(self._buy_tokens5000_button, 1)

        self._buy_tokens15000_button = QPushButton(_("Buy Tokens") + " (15000)", self)
        self._buy_tokens15000_button.clicked.connect(lambda: self._buy_tokens("15000"))
        buy_layout.addWidget(self._buy_tokens15000_button, 1)

        self._account_button = QPushButton(_("View Account"), self)
        self._account_button.setMinimumWidth(200)
        self._account_button.clicked.connect(self._view_account)
        layout.addWidget(self._account_button)

        self._logout_button = QPushButton(_("Sign out"), self)
        self._logout_button.setMinimumWidth(200)
        self._logout_button.clicked.connect(self._logout)
        layout.addWidget(self._logout_button)

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, user: User | None):
        if self._user is not user:
            Binding.disconnect_all(self._connections)
            self.setVisible(user is not None)

            self._user = user
            if user is not None:
                self._user_name.setText(user.name)
                self._connections = [
                    user.images_generated_changed.connect(self._update_counts),
                    user.credits_changed.connect(self._update_counts),
                ]
                self._update_counts()

    def _update_counts(self):
        user = util.ensure(self.user)
        self._images_generated.setText(str(user.images_generated))
        self._tokens_remaining.setText(str(user.credits))

    def _view_account(self):
        QDesktopServices.openUrl(QUrl(CloudClient.default_web_url + "/user"))

    def _buy_tokens(self, amount: str):
        QDesktopServices.openUrl(QUrl(f"{CloudClient.default_web_url}/checkout/tokens{amount}"))

    def _logout(self):
        eventloop.run(self._disconnect_and_logout())

    async def _disconnect_and_logout(self):
        await root.connection.disconnect()
        settings.access_token = ""
        settings.save()


class CloudWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 12, 4, 4)
        self.setLayout(layout)

        service_url = CloudClient.default_web_url
        service_url_text = (
            service_url.removeprefix("https://").removeprefix("www.").removesuffix("/")
        )
        header = QLabel(f"<b>{service_url_text}</b>", self)
        service_label = QLabel(f"<a href='{service_url}'>Visit Website</a>", self)
        service_label.setOpenExternalLinks(True)
        layout.addWidget(header)
        layout.addWidget(service_label)

        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._connection_status)

        self.connect_button = QPushButton(_("Login"), self)
        self.connect_button.setMinimumWidth(200)
        self.connect_button.setMinimumHeight(int(1.3 * self.connect_button.sizeHint().height()))
        self.connect_button.clicked.connect(self._connect)

        self._sign_out_button = QPushButton(_("Sign out"), self)
        self._sign_out_button.setVisible(False)
        self._sign_out_button.setMinimumWidth(200)
        self._sign_out_button.clicked.connect(self._sign_out)

        self._user_widget = UserWidget(self)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.connect_button)
        buttons_layout.addWidget(self._sign_out_button)

        connect_layout = QHBoxLayout()
        connect_layout.addLayout(buttons_layout)
        connect_layout.addWidget(self._user_widget)
        connect_layout.addStretch()
        layout.addLayout(connect_layout)

        layout.addStretch()

    def update_connection_state(self, state: ConnectionState):
        is_connected = state == ConnectionState.connected
        self.connect_button.setVisible(not is_connected)
        self._sign_out_button.setVisible(False)
        self._user_widget.user = root.connection.user

        if state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            self.connect_button.setText(_("Sign in"))
            self.connect_button.setEnabled(True)
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif state is ConnectionState.auth_pending:
            self.connect_button.setText(_("Sign in"))
            self.connect_button.setEnabled(False)
            self._connection_status.setText(_("Waiting for sign-in to complete..."))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connection_status.setVisible(True)
        elif state is ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
            self._user_widget.user = root.connection.user
        else:
            can_connect = state in [ConnectionState.disconnected, ConnectionState.error]
            self.connect_button.setEnabled(can_connect)
            self.connect_button.setText(_("Connect") if can_connect else _("Connected"))
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")

        if state in [ConnectionState.error, ConnectionState.auth_error]:
            error = root.connection.error or "Unknown error"
            self._connection_status.setText(f"<b>Error</b>: {error.removeprefix('Error: ')}")
            self._connection_status.setStyleSheet(f"color: {red}; font-weight:bold")
            self._connection_status.setVisible(True)
            if settings.access_token:
                self._sign_out_button.setVisible(True)

    def _connect(self):
        connection = root.connection
        if connection.state in [ConnectionState.auth_missing, ConnectionState.auth_error]:
            connection.sign_in()
        else:
            if client := connection.create_client(settings):
                connection.connect(client)

    def _sign_out(self):
        settings.access_token = ""
        settings.save()


_server_mode_text = {
    ServerMode.undefined: "Undefined",
    ServerMode.cloud: _("Online Service"),
    ServerMode.managed: _("Local Managed Server"),
    ServerMode.external: _("Custom Server"),
    ServerMode.novelai: _("NovelAI"),
}
_server_mode_status = {
    "signed_out": (_("Signed out"), grey),
    "not_installed": (_("Not installed"), grey),
    "not_running": (_("Not running"), grey),
    "not_connected": (_("Not connected"), grey),
    "connecting": (_("Connecting"), yellow),
    "connected": (_("Connected"), green),
    "error": (_("Error"), red),
}


class ServerModeButton(QPushButton):
    toggled = pyqtSignal(ServerMode)

    def __init__(self, mode: ServerMode, status: str, parent=None):
        self._text = _server_mode_text[mode]
        super().__init__(self._text, parent)
        self.mode = mode
        self._status = status
        self._is_checked = False

        font = QFontMetrics(self.font())
        self._text_width = font.horizontalAdvance(self._text)
        max_width = max(font.horizontalAdvance(s[0]) for s in _server_mode_status.values())
        self.setMinimumWidth(self._text_width + max_width + 32)
        self.setFixedHeight(int(1.3 * self.sizeHint().height()))

        self.clicked.connect(self._toggle)

    def _toggle(self):
        self.toggled.emit(self.mode)

    def setChecked(self, a0: bool):
        self._is_checked = a0
        self.update()

    def isChecked(self) -> bool:
        return self._is_checked

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, status: str):
        self._status = status
        self.update()

    def paintEvent(self, a0):
        status_text, color = _server_mode_status.get(self._status, (_("Unknown"), red))
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        style = util.ensure(self.style())
        if self.isChecked():
            opt.state |= QStyle.StateFlag.State_Sunken
        style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelButtonCommand, opt, painter, self)

        rect = self.rect().adjusted(8, 0, -8, 0)
        bold = self.font()
        bold.setBold(True)
        painter.setFont(bold)
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._text
        )
        painter.setPen(QColor(color))
        painter.setFont(self.font())
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, status_text
        )
        painter.end()


class ServerModeSelect(QWidget):
    changed = pyqtSignal(ServerMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._cloud_button = ServerModeButton(ServerMode.cloud, "signed_out", self)
        self._managed_button = ServerModeButton(ServerMode.managed, "not_installed", self)
        self._external_button = ServerModeButton(ServerMode.external, "not_connected", self)
        self._novelai_button = ServerModeButton(ServerMode.novelai, "not_connected", self)

        for button in (self._cloud_button, self._managed_button, self._external_button, self._novelai_button):
            button.toggled.connect(self._change_mode)

        layout.addWidget(self._cloud_button)
        layout.addWidget(self._managed_button)
        layout.addWidget(self._external_button)
        layout.addWidget(self._novelai_button)
        layout.addStretch()

    def _change_mode(self, mode: ServerMode):
        self.mode = mode
        self.changed.emit(mode)

    @property
    def mode(self):
        if self._cloud_button.isChecked():
            return ServerMode.cloud
        elif self._managed_button.isChecked():
            return ServerMode.managed
        elif self._external_button.isChecked():
            return ServerMode.external
        elif self._novelai_button.isChecked():
            return ServerMode.novelai
        return ServerMode.undefined

    @mode.setter
    def mode(self, mode: ServerMode):
        self._cloud_button.setChecked(mode is ServerMode.cloud)
        self._managed_button.setChecked(mode is ServerMode.managed)
        self._external_button.setChecked(mode is ServerMode.external)
        self._novelai_button.setChecked(mode is ServerMode.novelai)

    def update_status(self, state: ConnectionState, server_state: ServerState):
        self._cloud_button.status = "signed_out"
        self._external_button.status = "not_connected"
        self._novelai_button.status = "not_connected"
        match server_state:
            case ServerState.not_installed:
                self._managed_button.status = "not_installed"
            case ServerState.stopped:
                self._managed_button.status = "not_running"
            case _:
                self._managed_button.status = "not_connected"

        match self.mode, state, server_state:
            case ServerMode.cloud, ConnectionState.auth_missing | ConnectionState.auth_error, _:
                self._cloud_button.status = "signed_out"
            case ServerMode.cloud, ConnectionState.auth_pending, _:
                self._cloud_button.status = "connecting"
            case ServerMode.cloud, ConnectionState.connected, _:
                self._cloud_button.status = "connected"
            case ServerMode.cloud, ConnectionState.error, _:
                self._cloud_button.status = "error"
            case ServerMode.managed, _, ServerState.starting:
                self._managed_button.status = "connecting"
            case ServerMode.managed, ConnectionState.connecting, _:
                self._managed_button.status = "connecting"
            case ServerMode.managed, ConnectionState.connected, ServerState.running:
                self._managed_button.status = "connected"
            case ServerMode.managed, ConnectionState.error, _:
                self._managed_button.status = "error"
            case ServerMode.external, ConnectionState.disconnected, _:
                self._external_button.status = "disconnected"
            case ServerMode.external, ConnectionState.connecting, _:
                self._external_button.status = "connecting"
            case ServerMode.external, ConnectionState.connected, _:
                self._external_button.status = "connected"
            case ServerMode.external, ConnectionState.error, _:
                self._external_button.status = "error"
            case ServerMode.novelai, ConnectionState.connecting, _:
                self._novelai_button.status = "connecting"
            case ServerMode.novelai, ConnectionState.connected, _:
                self._novelai_button.status = "connected"
            case ServerMode.novelai, ConnectionState.error, _:
                self._novelai_button.status = "error"


class NovelAIConnectionWidget(QWidget):
    """NovelAI connection panel shown inside the Connection tab when NAI mode is selected."""

    value_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 12, 4, 4)
        self.setLayout(layout)

        header = QLabel("<b>NovelAI</b>", self)
        website_label = QLabel("<a href='https://novelai.net'>Visit NovelAI</a>", self)
        website_label.setOpenExternalLinks(True)
        layout.addWidget(header)
        layout.addWidget(website_label)

        # Token input
        add_header(layout, Settings._nai_api_token)
        token_layout = QHBoxLayout()
        self._token_edit = QLineEdit(self)
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("pst-...")
        self._token_edit.textChanged.connect(self._on_token_changed)
        token_layout.addWidget(self._token_edit)
        self._connect_button = QPushButton(_("Connect"), self)
        self._connect_button.clicked.connect(self._connect)
        token_layout.addWidget(self._connect_button)
        layout.addLayout(token_layout)

        # Connection status
        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._connection_status)

        # Subscription info area
        self._subscription_frame = QFrame(self)
        self._subscription_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self._subscription_frame.setLineWidth(2)
        self._subscription_frame.setVisible(False)
        sub_layout = QVBoxLayout()
        self._subscription_frame.setLayout(sub_layout)

        self._sub_tier = QLabel("", self)
        self._sub_tier.setStyleSheet("font-weight:bold")
        tier_layout = QHBoxLayout()
        tier_layout.addWidget(QLabel(_("Subscription:"), self), 0)
        tier_layout.addWidget(self._sub_tier, 1)
        sub_layout.addLayout(tier_layout)

        self._anlas_label = QLabel("", self)
        self._anlas_label.setStyleSheet("font-weight:bold")
        anlas_layout = QHBoxLayout()
        anlas_layout.addWidget(QLabel(_("Anlas:"), self), 0)
        anlas_layout.addWidget(self._anlas_label, 1)
        sub_layout.addLayout(anlas_layout)

        self._refresh_sub_button = QPushButton(_("Refresh"), self)
        self._refresh_sub_button.clicked.connect(self._refresh_subscription)
        sub_layout.addWidget(self._refresh_sub_button, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self._subscription_frame)
        layout.addStretch()

    def _on_token_changed(self, text: str):
        settings.nai_api_token = text
        settings.save()

    def _connect(self):
        settings.nai_api_token = self._token_edit.text()
        settings.save()
        if client := root.connection.create_client(settings):
            root.connection.connect(client)

    def _refresh_subscription(self):
        """Fetch subscription info from the NAI image API host."""
        eventloop.run(self._fetch_subscription())

    async def _fetch_subscription(self):
        try:
            from ..backend.network import RequestManager
            rm = RequestManager()
            token = settings.nai_api_token
            if not token:
                return
            # Official NAI serves user endpoints (/user/subscription etc.) on the
            # IMAGE API host. api.novelai.net answers 400 "update to the image URL"
            # for third-party tools. (Launcher: nai_api_endpoint.dart userUrl().)
            data = await rm.http(
                "GET", "https://image.novelai.net/user/subscription",
                timeout=15, bearer=token,
            )
            if isinstance(data, dict):
                tier_val = data.get("tier", 0)
                tier_names = {0: "Free", 1: "Tablet", 2: "Scroll", 3: "Opus"}
                self._sub_tier.setText(tier_names.get(tier_val, f"Tier {tier_val}"))

                # trainingStepsLeft contains Anlas info
                anlas = data.get("trainingStepsLeft", {})
                fixed = anlas.get("fixedTrainingStepsLeft", 0)
                purchased = anlas.get("purchasedTrainingSteps", 0)
                self._anlas_label.setText(str(fixed + purchased))
                self._subscription_frame.setVisible(True)
        except Exception as e:
            self._sub_tier.setText(_("Error fetching subscription"))
            self._anlas_label.setText(str(e))
            self._subscription_frame.setVisible(True)

    def update_connection_state(self, state: ConnectionState):
        self._token_edit.setText(settings.nai_api_token)
        is_connected = state == ConnectionState.connected

        if is_connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
            self._connect_button.setEnabled(True)
            self._connect_button.setText(_("Reconnect"))
            # Auto-fetch subscription on connect
            self._refresh_subscription()
        elif state is ConnectionState.connecting:
            self._connection_status.setText(_("Connecting..."))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connect_button.setEnabled(False)
        elif state is ConnectionState.error:
            error = root.connection.error or "Unknown error"
            self._connection_status.setText(f"<b>{_('Error')}</b>: {error.removeprefix('Error: ')}")
            self._connection_status.setStyleSheet(f"color: {red}; font-weight:bold")
            self._connect_button.setEnabled(True)
            self._connect_button.setText(_("Connect"))
        else:
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
            self._connect_button.setEnabled(True)
            self._connect_button.setText(_("Connect"))
            self._subscription_frame.setVisible(False)


class TokenEditDialog(QDialog):
    """Dialog for adding or editing a NovelAI API token entry."""

    def __init__(self, parent=None, name="", token=""):
        super().__init__(parent)
        self.setWindowTitle(_("Edit API Token"))
        self.setMinimumWidth(400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel(_("Name") + ":", self))
        self._name_edit = QLineEdit(self)
        self._name_edit.setText(name)
        self._name_edit.setPlaceholderText(_("e.g. Main Account"))
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel(_("API Token") + ":", self))
        self._token_edit = QLineEdit(self)
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setText(token)
        self._token_edit.setPlaceholderText("pst-...")
        layout.addWidget(self._token_edit)

        button_layout = QHBoxLayout()
        ok_button = QPushButton(_("OK"), self)
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton(_("Cancel"), self)
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    @property
    def name(self):
        return self._name_edit.text().strip()

    @property
    def token(self):
        return self._token_edit.text().strip()


class NovelAISettings(SettingsTab):
    """Independent NovelAI settings tab with multi-token management."""

    def __init__(self):
        super().__init__(_("NovelAI Settings"))

        # --- Visit NovelAI link ---
        header = QLabel("<b>NovelAI</b>", self)
        website_label = QLabel("<a href='https://novelai.net'>Visit NovelAI</a>", self)
        website_label.setOpenExternalLinks(True)
        self._layout.addWidget(header)
        self._layout.addWidget(website_label)
        self._layout.addSpacing(8)

        # --- Connection status ---
        self._connection_status = QLabel(self)
        self._connection_status.setWordWrap(True)
        self._connection_status.setTextFormat(Qt.TextFormat.RichText)
        self._layout.addWidget(self._connection_status)
        self._layout.addSpacing(8)

        # --- Subscription info ---
        self._subscription_frame = QFrame(self)
        self._subscription_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self._subscription_frame.setLineWidth(2)
        self._subscription_frame.setVisible(False)
        sub_layout = QVBoxLayout()
        self._subscription_frame.setLayout(sub_layout)

        self._sub_tier = QLabel("", self)
        self._sub_tier.setStyleSheet("font-weight:bold")
        tier_layout = QHBoxLayout()
        tier_layout.addWidget(QLabel(_("Subscription:"), self), 0)
        tier_layout.addWidget(self._sub_tier, 1)
        sub_layout.addLayout(tier_layout)

        self._anlas_label = QLabel("", self)
        self._anlas_label.setStyleSheet("font-weight:bold")
        anlas_layout = QHBoxLayout()
        anlas_layout.addWidget(QLabel(_("Anlas:"), self), 0)
        anlas_layout.addWidget(self._anlas_label, 1)
        sub_layout.addLayout(anlas_layout)

        self._refresh_sub_button = QPushButton(_("Refresh"), self)
        self._refresh_sub_button.clicked.connect(self._refresh_subscription)
        sub_layout.addWidget(self._refresh_sub_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self._layout.addWidget(self._subscription_frame)
        self._layout.addSpacing(12)

        # --- API Token management ---
        token_header = QLabel("<b>" + _("API Token Management") + "</b>", self)
        self._layout.addWidget(token_header)

        # Current token selector
        current_label = QLabel(_("Active Token:"), self)
        self._token_combo = QComboBox(self)
        self._token_combo.setMinimumWidth(250)
        self._token_combo.currentIndexChanged.connect(self._on_active_token_changed)
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(current_label)
        combo_layout.addWidget(self._token_combo, 1)
        self._layout.addLayout(combo_layout)

        # Token list
        self._token_list = QListWidget(self)
        self._token_list.setMaximumHeight(150)
        self._layout.addWidget(self._token_list)

        # Buttons
        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton(_("Add"), self)
        self._add_btn.clicked.connect(self._add_token)
        self._edit_btn = QPushButton(_("Edit"), self)
        self._edit_btn.clicked.connect(self._edit_token)
        self._delete_btn = QPushButton(_("Delete"), self)
        self._delete_btn.clicked.connect(self._delete_token)
        btn_layout.addWidget(self._add_btn)
        btn_layout.addWidget(self._edit_btn)
        btn_layout.addWidget(self._delete_btn)
        btn_layout.addStretch()
        self._layout.addLayout(btn_layout)

        self._layout.addStretch()

        root.connection.state_changed.connect(self._update_connection_state)

    def _read(self):
        self._refresh_token_list()
        self._update_connection_state(root.connection.state)

    def _write(self):
        pass  # Token management is handled via explicit button actions

    def _refresh_token_list(self):
        """Rebuild the token list and combo from settings."""
        tokens = settings.nai_tokens
        active_idx = settings.nai_active_token_index

        # Block signals to avoid triggering _on_active_token_changed during rebuild
        self._token_combo.blockSignals(True)
        self._token_combo.clear()
        self._token_list.clear()

        if not tokens:
            self._token_combo.addItem(_("(No tokens configured)"))
            self._token_combo.blockSignals(False)
            return

        for i, entry in enumerate(tokens):
            name = entry.get("name", f"Token {i+1}")
            tier = entry.get("tier", "")
            anlas = entry.get("anlas", 0)
            display = name
            if tier:
                display += f"  [{tier}]"
            if anlas:
                display += f"  ({anlas} Anlas)"

            self._token_combo.addItem(display)
            item = QListWidgetItem(display)
            self._token_list.addItem(item)

        safe_idx = max(0, min(active_idx, len(tokens) - 1))
        self._token_combo.setCurrentIndex(safe_idx)
        self._token_list.setCurrentRow(safe_idx)
        self._token_combo.blockSignals(False)

    def _on_active_token_changed(self, index: int):
        if index < 0:
            return
        tokens = settings.nai_tokens
        if 0 <= index < len(tokens):
            settings.set_active_nai_token_index(index)
            settings.save()

    def _add_token(self):
        dlg = TokenEditDialog(self)
        if dlg.exec_() == QDialog.DialogCode.Accepted and dlg.token:
            tokens = list(settings.nai_tokens)
            new_entry = {
                "name": dlg.name or f"Token {len(tokens)+1}",
                "token": dlg.token,
                "tier": "",
                "anlas": 0,
            }
            tokens.append(new_entry)
            settings.nai_tokens = tokens
            settings.set_active_nai_token_index(len(tokens) - 1)
            settings.save()
            self._refresh_token_list()

    def _edit_token(self):
        tokens = list(settings.nai_tokens)
        idx = self._token_list.currentRow()
        if 0 <= idx < len(tokens):
            entry = tokens[idx]
            dlg = TokenEditDialog(self, entry.get("name", ""), entry.get("token", ""))
            if dlg.exec_() == QDialog.DialogCode.Accepted and dlg.token:
                tokens[idx] = {
                    "name": dlg.name or entry.get("name", ""),
                    "token": dlg.token,
                    "tier": entry.get("tier", ""),
                    "anlas": entry.get("anlas", 0),
                }
                settings.nai_tokens = tokens
                # If editing the active token, sync nai_api_token
                if idx == settings.nai_active_token_index:
                    settings.nai_api_token = dlg.token
                settings.save()
                self._refresh_token_list()

    def _delete_token(self):
        tokens = list(settings.nai_tokens)
        idx = self._token_list.currentRow()
        if 0 <= idx < len(tokens):
            confirm = QMessageBox.question(
                self,
                _("Delete Token"),
                _("Are you sure you want to delete this token?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                tokens.pop(idx)
                settings.nai_tokens = tokens
                # Adjust active index
                if len(tokens) == 0:
                    settings.nai_active_token_index = 0
                    settings.nai_api_token = ""
                elif idx <= settings.nai_active_token_index:
                    new_idx = max(0, settings.nai_active_token_index - 1)
                    settings.set_active_nai_token_index(new_idx)
                settings.save()
                self._refresh_token_list()

    def _refresh_subscription(self):
        """Fetch subscription info for the active token."""
        eventloop.run(self._fetch_subscription())

    async def _fetch_subscription(self):
        try:
            from ..backend.network import RequestManager
            rm = RequestManager()
            token = settings.get_active_nai_token()
            if not token:
                return
            # User endpoints live on the image API host (see note above).
            data = await rm.http(
                "GET", "https://image.novelai.net/user/subscription",
                timeout=15, bearer=token,
            )
            if isinstance(data, dict):
                tier_val = data.get("tier", 0)
                tier_names = {0: "Free", 1: "Tablet", 2: "Scroll", 3: "Opus"}
                tier_str = tier_names.get(tier_val, f"Tier {tier_val}")
                self._sub_tier.setText(tier_str)

                anlas = data.get("trainingStepsLeft", {})
                fixed = anlas.get("fixedTrainingStepsLeft", 0)
                purchased = anlas.get("purchasedTrainingSteps", 0)
                anlas_total = fixed + purchased
                self._anlas_label.setText(str(anlas_total))
                self._subscription_frame.setVisible(True)

                # Update token entry with tier/anlas info
                tokens = list(settings.nai_tokens)
                idx = settings.nai_active_token_index
                if 0 <= idx < len(tokens):
                    tokens[idx]["tier"] = tier_str
                    tokens[idx]["anlas"] = anlas_total
                    settings.nai_tokens = tokens
                    settings.save()
                    self._refresh_token_list()
        except Exception as e:
            self._sub_tier.setText(_("Error fetching subscription"))
            self._anlas_label.setText(str(e))
            self._subscription_frame.setVisible(True)

    def _update_connection_state(self, state=None):
        if state is None:
            state = root.connection.state
        is_nai = settings.server_mode is ServerMode.novelai
        if not is_nai:
            self._connection_status.setText(_("Not using NovelAI backend"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
            self._subscription_frame.setVisible(False)
            return

        if state == ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
            self._refresh_subscription()
        elif state == ConnectionState.connecting:
            self._connection_status.setText(_("Connecting..."))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
        elif state == ConnectionState.error:
            error = root.connection.error or "Unknown error"
            self._connection_status.setText(f"<b>{_('Error')}</b>: {error.removeprefix('Error: ')}")
            self._connection_status.setStyleSheet(f"color: {red}; font-weight:bold")
        else:
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
            self._subscription_frame.setVisible(False)


class ConnectionSettings(SettingsTab):
    def __init__(self, server: Server):
        super().__init__(_("Server Configuration"))
        self._server = server

        self._server_mode = ServerModeSelect(self)
        self._server_mode.changed.connect(self._change_server_mode)

        self._setup_widget = InitialSetupWidget(self)
        self._cloud_widget = CloudWidget(self)
        self._server_widget = ServerWidget(server, self)
        self._connection_widget = QWidget(self)
        self._novelai_widget = NovelAIConnectionWidget(self)
        self._server_stack = QStackedWidget(self)
        self._server_stack.addWidget(self._setup_widget)
        self._server_stack.addWidget(self._cloud_widget)
        self._server_stack.addWidget(self._server_widget)
        self._server_stack.addWidget(self._connection_widget)
        self._server_stack.addWidget(self._novelai_widget)

        connection_layout = QVBoxLayout()
        connection_layout.setContentsMargins(0, 0, 0, 0)
        self._connection_widget.setLayout(connection_layout)

        add_header(connection_layout, Settings._server_url)
        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self._connection_widget)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton(_("Connect"), self._connection_widget)
        self._connect_button.clicked.connect(self._connect)
        server_layout.addWidget(self._connect_button)
        connection_layout.addLayout(server_layout)

        self._connection_status = QLabel(self._connection_widget)
        self._supported_workloads = QLabel(self._connection_widget)
        self._supported_workloads.setWordWrap(True)
        self._supported_workloads.setTextFormat(Qt.TextFormat.RichText)
        self._supported_workloads.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._supported_workloads.setOpenExternalLinks(True)

        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._connection_status)
        status_layout.addWidget(open_log_button, alignment=Qt.AlignmentFlag.AlignRight)

        connection_layout.addLayout(status_layout)
        connection_layout.addWidget(self._supported_workloads)
        connection_layout.addStretch()

        self._layout.addWidget(self._server_mode)
        self._layout.addWidget(self._server_stack)

        self.update_server_status()
        self._update_server_mode(settings.server_mode)

        root.connection.state_changed.connect(self.update_server_status)
        root.connection.error_changed.connect(self.update_server_status)
        root.connection.progress_changed.connect(self.update_server_status)
        self._setup_widget.finished.connect(self._setup_finished)
        self._server_widget.state_changed.connect(self.update_server_status)

    def _setup_finished(self, mode: ServerMode):
        self._server_mode.mode = mode
        self._update_server_mode(mode)

    def _update_server_mode(self, mode: ServerMode):
        self._server_mode.setVisible(mode is not ServerMode.undefined)
        widget = {
            ServerMode.cloud: self._cloud_widget,
            ServerMode.managed: self._server_widget,
            ServerMode.external: self._connection_widget,
            ServerMode.novelai: self._novelai_widget,
            ServerMode.undefined: self._setup_widget,
        }[mode]
        self._server_stack.setCurrentWidget(widget)

    def update_ui(self):
        self._server_widget.update_ui()

    def _read(self):
        self._server_mode.mode = settings.server_mode
        self._server_mode.update_status(root.connection.state, self._server.state)
        self._update_server_mode(settings.server_mode)
        self._server_url.setText(settings.server_url)

    def _write(self):
        settings.server_mode = self._server_mode.mode
        settings.server_url = self._server_url.text()

    def _change_server_mode(self):
        self._update_server_mode(self._server_mode.mode)
        self.write()

    def _connect(self):
        if client := root.connection.create_client(settings):
            root.connection.connect(client)

    def update_server_status(self):
        connection = root.connection
        self._server_mode.update_status(connection.state, self._server.state)
        self._cloud_widget.update_connection_state(connection.state)
        if settings.server_mode is ServerMode.novelai:
            self._novelai_widget.update_connection_state(connection.state)
        self._connect_button.setEnabled(True)
        if connection.state == ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
        elif connection.state == ConnectionState.connecting:
            self._connection_status.setText(_("Connecting"))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connect_button.setEnabled(False)
        elif connection.state == ConnectionState.discover_models:
            progress = f" ({connection.progress[0]}/{connection.progress[1]})"
            self._connection_status.setText(_("Discovering models") + progress)
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connect_button.setEnabled(False)
        elif connection.state == ConnectionState.disconnected:
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif connection.state == ConnectionState.error:
            msg = connection.error.removeprefix("Error: ") if connection.error else "Unknown error"
            self._connection_status.setText("<b>" + _("Error") + f"</b>: {msg}")
            self._connection_status.setStyleSheet(f"color: {red};")

        self._supported_workloads.clear()
        if connection.state in [ConnectionState.connected, ConnectionState.error]:
            if connection.missing_resources is not None:
                self._show_missing_resources(connection.missing_resources, connection.state)

    def _show_missing_resources(self, res: MissingResources, state: ConnectionState):
        def model_name(id: ResourceId, with_file=False):
            if res := resources.find_resource(id):
                if with_file:
                    return f"{res.name} ({', '.join(f.name for f in res.files)})"
                return res.name
            if isinstance(id.identifier, str):
                return f"{id.kind.value} {id.identifier}"
            return f"{id.kind.value} {id.identifier.value}"

        text = ""
        if isinstance(res.missing, list):
            text = (
                _("The following ComfyUI custom nodes are missing or too old")
                + ":<ul>"
                + "\n".join(f"<li>{p.name} <a href='{p.url}'>{p.url}</a></li>" for p in res.missing)
                + "</ul>"
                + _(
                    "Please install or update the custom node package, then restart the server and try again."
                )
                + _("If nodes are still missing, check the ComfyUI output at startup for errors.")
                + "<br>"
            )
        else:
            basic = [m for lst in res.missing.values() for m in lst if m.arch is Arch.all]
            basic = util.unique(basic, key=lambda m: m.string)
            if len(basic) > 0:
                text = _("Missing common models (required)") + ":\n<ul>"
                text += "\n".join(f"<li>{model_name(m, True)}</li>" for m in basic)
                text += "</ul>"
            text += _("Detected workloads for the following base models:") + "\n<ul>"
            for arch, missing in res.missing.items():
                if arch in [Arch.all, Arch.illu_v]:
                    continue
                text += f"<li><b>{arch.value}</b>: "
                if len(missing) == 0:
                    text += _("supported")
                else:
                    names = [model_name(m) for m in missing if m.arch is arch]
                    if len(names) > 0:
                        text += _("missing") + " " + ", ".join(names)
                    else:
                        text += _("models found")
                text += "</li>"
            text += "</ul>"

        link = "<a href='https://docs.interstice.cloud/comfyui-setup'>Custom ComfyUI Setup</a>"
        text += _(
            "See {link} for required models.<br>Check the client.log file for more details.",
            link=link,
        )
        style = "" if state is ConnectionState.error else f"color: {grey};"
        self._supported_workloads.setStyleSheet(style)
        self._supported_workloads.setText(text)

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


class DiffusionSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Diffusion Settings"))

        S = Settings
        self.add("selection_feather", SliderSetting(S._selection_feather, self, 0, 25, "{} %"))
        self.add("selection_blend", SliderSetting(S._selection_blend, self, 0, 100, "{} px"))
        self.add("selection_padding", SliderSetting(S._selection_padding, self, 0, 25, "{} %"))
        self.add("color_match", SwitchSetting(S._color_match, parent=self))
        self.add("nsfw_filter", ComboBoxSetting(S._nsfw_filter, parent=self))

        nsfw_settings = [(_("Disabled"), 0.0), (_("Basic"), 0.65), (_("Strict"), 0.8)]
        self._widgets["nsfw_filter"].set_items(nsfw_settings)
        DiffusionSettings._warning_shown = self._warning_shown or settings.nsfw_filter > 0

        self._layout.addStretch()

    _warning_shown = False

    def _write(self):
        if self._widgets["nsfw_filter"].value > 0 and not self._warning_shown:
            DiffusionSettings._warning_shown = True
            QMessageBox.warning(
                self,
                _("NSFW Filter Warning"),
                _(
                    "The NSFW filter is a basic tool to exclude explicit content from generated images. It is NOT a guarantee and may not catch all inappropriate content. Please use responsibly and always review the generated images."
                ),
            )


class InterfaceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Interface Settings"))

        S = Settings
        self.add("language", ComboBoxSetting(S._language, parent=self))
        self.add("prompt_translation", ComboBoxSetting(S._prompt_translation, parent=self))
        self.add(
            "prompt_line_count",
            SpinBoxSetting(S._prompt_line_count, self, 1, prompt_max_line_count),
        )
        self.add(
            "show_negative_prompt",
            SwitchSetting(S._show_negative_prompt, (_("Show"), _("Hide")), self),
        )
        self.add("show_steps", SwitchSetting(S._show_steps, parent=self))
        self.add("recent_styles_count", SpinBoxSetting(S._recent_styles_count, self, 0, 10))

        self.add("tag_files", FileListSetting(S._tag_files, files=self._tag_files(), parent=self))
        self._layout.addWidget(self._widgets["tag_files"].list_widget)
        self._widgets["tag_files"].add_button(
            Krita.instance().icon("reload-preset"),
            _("Look for new tag files"),
            self._update_tag_files,
        )
        self._widgets["tag_files"].add_button(
            Krita.instance().icon("document-open"),
            _("Open folder where custom tag files can be placed"),
            self._open_tag_folder,
        )

        self.add(
            "generation_finished_action",
            ComboBoxSetting(S._generation_finished_action, parent=self),
        )
        self.add("apply_behavior", ComboBoxSetting(S._apply_behavior, parent=self))
        self.add("apply_region_behavior", ComboBoxSetting(S._apply_region_behavior, parent=self))
        self.add("apply_behavior_live", ComboBoxSetting(S._apply_behavior_live, parent=self))
        self.add(
            "apply_region_behavior_live",
            ComboBoxSetting(S._apply_region_behavior_live, parent=self),
        )
        self.add("new_seed_after_apply", SwitchSetting(S._new_seed_after_apply, parent=self))
        self.add("save_image_format", ComboBoxSetting(S._save_image_format, parent=self))
        self.add("save_image_metadata", SwitchSetting(S._save_image_metadata, parent=self))
        self.add("debug_dump_workflow", SwitchSetting(S._debug_dump_workflow, parent=self))

        self._widgets["save_image_format"].value_changed.connect(self._update_image_format_widgets)

        languages = [(lang.name, lang.id) for lang in Localization.available]
        self._widgets["language"].set_items(languages)
        self.update_translation(root.connection.client_if_connected)

        for w in ["apply_region_behavior", "apply_region_behavior_live"]:
            self._widgets[w].show_label = False

        self._layout.addStretch()

    def read(self):
        super().read()
        self._update_image_format_widgets()

    def _tag_files(self) -> list[str]:
        plugin_tags_path = util.plugin_dir / "tags"
        user_tags_path = util.user_data_dir / "tags"
        files = set()
        for path in plugin_tags_path.glob("*.csv"):
            files.add(path.stem)
        for path in user_tags_path.glob("*.csv"):
            files.add(path.stem)

        return list(files)

    def _update_tag_files(self):
        self._widgets["tag_files"].reset_files(self._tag_files())

    def _open_tag_folder(self):
        user_tag_folder = util.user_data_dir / "tags"
        user_tag_folder.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_tag_folder)))

    def update_translation(self, client: Client | None):
        translation: ComboBoxSetting = self._widgets["prompt_translation"]
        languages = [("Disabled", "")]
        if client:
            languages += [(lang.name, lang.code) for lang in client.features.languages]
        translation.enabled = client is not None
        translation.set_items(languages)
        self.read()

    def _update_image_format_widgets(self):
        fmt: ImageFileFormat = self._widgets["save_image_format"].value
        self._widgets["save_image_metadata"].enabled = fmt.extension == "png"


class HistorySizeWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, maximum: int, step: int, parent=None):
        super().__init__(parent)

        self._history_size = QSpinBox(self)
        self._history_size.setMinimum(5)
        self._history_size.setMaximum(maximum)
        self._history_size.setSingleStep(step)
        self._history_size.setSuffix(" MB")
        self._history_size.valueChanged.connect(self._change_value)

        self._history_usage = QLabel(self)
        self._history_usage.setStyleSheet(f"font-style:italic; color: {green};")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._history_size)
        layout.addWidget(self._history_usage)
        self.setLayout(layout)

    def _change_value(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._history_size.value()

    @value.setter
    def value(self, v):
        self._history_size.setValue(v)

    def update_usage(self, usage: float):
        self._history_usage.setText(_("Currently using") + f" {usage:.1f} MB")


class PerformanceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Performance Settings"))

        add_header(self._layout, Settings._history_size)
        self._history_size = HistorySizeWidget(maximum=20000, step=100, parent=self)
        self._history_size.value_changed.connect(self.write)
        self._layout.addWidget(self._history_size)

        add_header(self._layout, Settings._history_storage)
        self._history_storage = HistorySizeWidget(maximum=2000, step=5, parent=self)
        self._history_storage.value_changed.connect(self.write)
        self._layout.addWidget(self._history_storage)

        add_header(self._layout, Settings._performance_preset)
        self._device_info = QLabel(self)
        self._device_info.setStyleSheet("font-style:italic")
        self._layout.addWidget(self._device_info)

        self._performance_preset = QComboBox(self)
        for preset in PerformancePreset:
            self._performance_preset.addItem(preset.value)
        self._performance_preset.currentIndexChanged.connect(self._change_performance_preset)
        self._layout.addWidget(self._performance_preset, alignment=Qt.AlignmentFlag.AlignLeft)

        self._advanced = QWidget(self)
        self._advanced.setEnabled(settings.performance_preset is PerformancePreset.custom)
        self._layout.addWidget(self._advanced)
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(8, 0, 0, 4)
        self._advanced.setLayout(advanced_layout)

        self._batch_size = SliderSetting(Settings._batch_size, self._advanced, 1, 16)
        self._batch_size.value_changed.connect(self.write)
        advanced_layout.addWidget(self._batch_size)

        self._resolution_multiplier = SliderSetting(
            Settings._resolution_multiplier, self._advanced, 0.3, 1.5, "{:.1f}x"
        )
        self._resolution_multiplier.value_changed.connect(self.write)
        advanced_layout.addWidget(self._resolution_multiplier)

        self._max_pixel_count = SpinBoxSetting(
            Settings._max_pixel_count, self._advanced, 1, 99, 1, " MP"
        )
        self._max_pixel_count.value_changed.connect(self.write)
        advanced_layout.addWidget(self._max_pixel_count)

        self._tiled_vae = SwitchSetting(
            Settings._tiled_vae, text=(_("Always"), _("Automatic")), parent=self._advanced
        )
        self._tiled_vae.value_changed.connect(self.write)
        advanced_layout.addWidget(self._tiled_vae)

        self._dynamic_caching = SwitchSetting(Settings._dynamic_caching, parent=self)
        self._dynamic_caching.value_changed.connect(self.write)
        self._layout.addWidget(self._dynamic_caching)

        self._multi_threading = SwitchSetting(Settings._multi_threading, parent=self)
        self._multi_threading.value_changed.connect(self.write)
        self._layout.addWidget(self._multi_threading)

        self._layout.addStretch()

    def _change_performance_preset(self, index):
        self.write()
        is_custom = settings.performance_preset is PerformancePreset.custom
        self._advanced.setEnabled(is_custom)
        if (
            settings.performance_preset is PerformancePreset.auto
            and root.connection.state is ConnectionState.connected
        ):
            apply_performance_preset(settings, root.connection.client.device_info)
        if not is_custom:
            self.read()

    def update_client_info(self):
        if root.connection.state is ConnectionState.connected:
            client = root.connection.client
            self._device_info.setText(
                _("Device")
                + f": [{client.device_info.type.upper()}] {client.device_info.name} ({client.device_info.vram} GB)"
            )

    def _read(self):
        self._history_size.value = settings.history_size
        self._history_size.update_usage(root.active_model.jobs.memory_usage)
        self._history_storage.value = settings.history_storage
        self._history_storage.update_usage(root.get_active_model_used_storage() / (1024**2))
        self._multi_threading.value = settings.multi_threading
        self._batch_size.value = settings.batch_size
        self._performance_preset.setCurrentIndex(
            list(PerformancePreset).index(settings.performance_preset)
        )
        self._resolution_multiplier.value = settings.resolution_multiplier
        self._max_pixel_count.value = settings.max_pixel_count
        self._tiled_vae.value = settings.tiled_vae
        self._dynamic_caching.value = settings.dynamic_caching
        self.update_client_info()

    def _write(self):
        settings.history_size = self._history_size.value
        settings.history_storage = self._history_storage.value
        settings.multi_threading = self._multi_threading.value
        settings.batch_size = int(self._batch_size.value)
        settings.resolution_multiplier = self._resolution_multiplier.value
        settings.max_pixel_count = self._max_pixel_count.value
        settings.tiled_vae = self._tiled_vae.value
        settings.performance_preset = list(PerformancePreset)[
            self._performance_preset.currentIndex()
        ]
        settings.dynamic_caching = self._dynamic_caching.value


class AboutSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Plugin Information and Updates"))

        large = self.font()
        large.setPointSize(large.pointSize() + 2)

        extra_large = self.font()
        extra_large.setPointSize(extra_large.pointSize() + 4)

        bold = self.font()
        bold.setBold(True)

        italic = self.font()
        italic.setItalic(True)

        header_layout = QHBoxLayout()
        header_logo = QLabel(self)
        font_height = QFontMetrics(extra_large).height() + 4
        header_logo.setPixmap(logo().scaled(font_height * 2, font_height * 2))
        header_logo.setMaximumSize(font_height * 2, font_height * 2)
        header_text = QLabel("Generative AI\nfor Krita", self)
        header_text.setFont(extra_large)
        header_layout.addWidget(header_logo)
        header_layout.addWidget(header_text)

        current_version_name = QLabel(_("Current version") + ":", self)
        current_version_value = QLabel(__version__, self)

        latest_version_name = QLabel(_("Latest version") + ":", self)
        self._latest_version_value = QLabel(self)
        self._latest_version_value.setFont(bold)

        self._update_error = QLabel(self)
        self._update_error.setFont(italic)

        self._update_checkbox = QCheckBox(_("Check for updates on startup"), self)
        self._update_checkbox.setChecked(settings.auto_update)
        self._update_checkbox.stateChanged.connect(self.write)

        self._check_button = QPushButton(_("Check for Updates"), self)
        self._check_button.setMinimumWidth(font_height * 6)
        self._check_button.clicked.connect(self._check_updates)

        self._update_button = QPushButton(_("Download and Install"), self)
        self._update_button.setMinimumWidth(font_height * 6)
        self._update_button.clicked.connect(self._run_update)

        sys_header = QLabel(_("System Information"), self)
        sys_header.setFont(large)
        sys_desc = QLabel(_("Please attach this information when reporting issues!"), self)
        sys_desc.setFont(italic)
        sys_button = QPushButton(_("Collect Diagnostics"), self)
        sys_button.setMinimumWidth(font_height * 6)
        sys_button.clicked.connect(self._collect_diagnostics)
        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        doc_header = QLabel(_("Documentation and Support"), self)
        doc_header.setFont(large)

        doc_links = QLabel(_links_text, self)
        doc_links.setOpenExternalLinks(True)
        doc_contact = QLabel(_contact_text, self)
        doc_contact.setOpenExternalLinks(True)

        self._layout.addLayout(header_layout)
        self._layout.addSpacing(10)
        current_version_layout = QHBoxLayout()
        current_version_layout.addWidget(current_version_name)
        current_version_layout.addWidget(current_version_value)
        current_version_layout.addStretch()
        self._layout.addLayout(current_version_layout)
        latest_version_layout = QHBoxLayout()
        latest_version_layout.addWidget(latest_version_name)
        latest_version_layout.addWidget(self._latest_version_value)
        latest_version_layout.addStretch()
        self._layout.addLayout(latest_version_layout)
        self._layout.addWidget(self._update_error)
        self._layout.addWidget(self._update_checkbox)
        update_layout = QHBoxLayout()
        update_layout.addWidget(self._check_button)
        update_layout.addWidget(self._update_button)
        update_layout.addStretch()
        self._layout.addLayout(update_layout)
        self._layout.addSpacing(20)
        self._layout.addWidget(sys_header)
        self._layout.addWidget(sys_desc)
        self._layout.addWidget(sys_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self._layout.addWidget(open_log_button)
        self._layout.addSpacing(20)
        self._layout.addWidget(doc_header)
        self._layout.addSpacing(5)
        doc_layout = QHBoxLayout()
        doc_layout.addWidget(doc_links)
        doc_layout.addSpacing(40)
        doc_layout.addWidget(doc_contact)
        doc_layout.addStretch()
        self._layout.addLayout(doc_layout)
        self._layout.addStretch()

        root.auto_update.state_changed.connect(self._update_content)
        self._update_content()

    def _update_content(self):
        self._check_button.setEnabled(False)
        self._update_button.setEnabled(False)
        self._update_error.clear()

        au = root.auto_update
        match au.state:
            case UpdateState.unknown:
                self._latest_version_value.setText(_("Not checked"))
                self._check_button.setEnabled(True)
            case UpdateState.checking:
                self._latest_version_value.setText(_("Checking for updates..."))
            case UpdateState.latest:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
            case UpdateState.available:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.downloading:
                self._latest_version_value.setText(_("Downloading package..."))
            case UpdateState.installing:
                self._latest_version_value.setText(_("Installing new version..."))
            case UpdateState.failed_check:
                self._latest_version_value.setText(_("Unknown"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
            case UpdateState.failed_update:
                self._latest_version_value.setText(_("Update failed"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.restart_required:
                self._latest_version_value.setText(
                    _("Please restart Krita to complete the update!")
                )

    def _check_updates(self):
        root.auto_update.check()

    def _run_update(self):
        root.auto_update.run()

    def _read(self):
        self._update_checkbox.setChecked(settings.auto_update)

    def _write(self):
        settings.auto_update = self._update_checkbox.isChecked()

    def _collect_diagnostics(self):
        diagnostics = collect_diagnostics()
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(diagnostics)

        window = QDialog(self)
        window.setWindowTitle(_("Diagnostics Information"))
        layout = QVBoxLayout()
        text = QTextEdit(window)
        text.setReadOnly(True)
        text.setText(diagnostics)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.SmallestReadableFont))
        if clipboard is not None:
            msg = _("System information has been copied to the clipboard.")
            layout.addWidget(QLabel("✔️ " + msg))
        layout.addSpacing(6)
        layout.addWidget(text)
        window.setLayout(layout)
        window.resize(min(self.width(), 800), 640)
        window.exec_()

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


_links_text = """
<a href='https://www.interstice.cloud'>Website</a><br><br>
<a href='https://docs.interstice.cloud'>Handbook: Guides and Tips</a><br><br>
<a href='https://github.com/Acly/krita-ai-diffusion'>GitHub</a>
"""

_contact_text = """
<a href='https://github.com/Acly/krita-ai-diffusion/issues'>Issues</a><br><br>
<a href='https://github.com/Acly/krita-ai-diffusion/discussions'>Discussions</a><br><br>
<a href='https://discord.gg/pWyzHfHHhU'>Discord</a>
"""


class SettingsDialog(QDialog):
    _instance = None

    @classmethod
    def instance(cls) -> SettingsDialog:
        assert cls._instance is not None
        return cls._instance

    def __init__(self, server: Server):
        super().__init__()
        type(self)._instance = self

        self.setWindowTitle(_("Configure Image Diffusion"))
        self.setMinimumSize(QSize(960, 480))
        if screen := QGuiApplication.screenAt(QCursor.pos()):
            size = screen.availableSize()
            min_w = min(size.width(), QFontMetrics(self.font()).width("M") * 100)
            self.resize(QSize(min_w, int(size.height() * 0.8)))

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.connection = ConnectionSettings(server)
        self.styles = StylePresets(server)
        self.novelai_styles = StylePresets(server, nai_only=True)
        self.novelai = NovelAISettings()
        self.diffusion = DiffusionSettings()
        self.interface = InterfaceSettings()
        self.performance = PerformanceSettings()
        self.about = AboutSettings()

        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setFixedWidth(120)

        def create_list_item(text: str, widget: QWidget):
            item = QListWidgetItem(text, self._list)
            item.setSizeHint(QSize(112, 24))
            self._stack.addWidget(widget)

        create_list_item(_("Connection"), self.connection)
        create_list_item(_("Styles"), self.styles)
        create_list_item(_("NovelAI 风格"), self.novelai_styles)
        create_list_item(_("NovelAI"), self.novelai)
        create_list_item(_("Diffusion"), self.diffusion)
        create_list_item(_("Interface"), self.interface)
        create_list_item(_("Performance"), self.performance)
        create_list_item(_("Plugin"), self.about)

        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._change_page)
        layout.addWidget(self._list)

        inner = QVBoxLayout()
        layout.addLayout(inner)
        inner.addWidget(self._stack)
        inner.addSpacing(6)

        self._restore_button = QPushButton(_("Restore Defaults"), self)
        self._restore_button.clicked.connect(self.restore_defaults)

        version_label = QLabel(_("Plugin version") + f": {__version__}", self)
        version_label.setStyleSheet(f"font-style:italic; color: {grey};")

        anchor = _("Open Settings folder")
        self._open_folder_link = QLabel(f"<a href='file://{util.user_data_dir}'>{anchor}</a>", self)
        self._open_folder_link.linkActivated.connect(self._open_settings_folder)
        self._open_folder_link.setToolTip(str(util.user_data_dir))

        self._close_button = QPushButton(_("Ok"), self)
        self._close_button.clicked.connect(self._close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(version_label)
        button_layout.addStretch()
        button_layout.addWidget(self._open_folder_link)
        button_layout.addSpacing(8)
        button_layout.addWidget(self._close_button)
        inner.addLayout(button_layout)

        root.connection.state_changed.connect(self._update_connection)
        root.connection.models_changed.connect(self.styles.update_model_lists)
        root.connection.models_changed.connect(self.novelai_styles.update_model_lists)

    def read(self):
        self.connection.read()
        self.styles.read()
        self.novelai_styles.read()
        self.novelai.read()
        self.diffusion.read()
        self.interface.read()
        self.performance.read()
        self.about.read()

    def restore_defaults(self):
        settings.restore()
        settings.save()
        self.read()

    def show(self, style: Style | None = None):
        self.read()
        self.connection.update_ui()
        super().show()

        if style:
            if is_nai_style(style):
                self._list.setCurrentRow(2)  # NovelAI 风格 page
                self.novelai_styles.current_style = style
            else:
                self._list.setCurrentRow(1)  # Styles page
                self.styles.current_style = style
        self._close_button.setFocus()

    def _change_page(self, index):
        self._stack.setCurrentIndex(index)

    def _update_connection(self):
        self.connection.update_server_status()
        if root.connection.state is ConnectionState.connected:
            self.interface.update_translation(root.connection.client)
            self.performance.update_client_info()

    def _open_settings_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.user_data_dir)))

    def _close(self):
        _ = self.close()
