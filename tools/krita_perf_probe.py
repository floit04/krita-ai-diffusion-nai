"""Krita 主线程卡顿探针 (probe).

用途：找出"大分辨率下移动/旋转特别卡"到底卡在哪一层。

它测三件事：
  1. 主线程事件循环延迟 —— 一个 4ms 的 QTimer，实际间隔减去 4ms 就是主线程
     被占住的时间。这是"卡"的直接定义，不依赖任何猜测。
  2. 卡住的瞬间主线程在跑什么 Python 代码 —— 后台采样线程在检测到主线程
     超过阈值没有心跳时，抓 sys._current_frames() 的主线程栈。
     如果抓到的是 <no python frame>，说明卡在 Krita 的 C++ 里，与 Python 插件无关；
     如果抓到 ai_diffusion 里的帧，说明是本插件的轮询在偷主线程。
  3. 本插件几个轮询回调的调用次数/总耗时/最大单次耗时（打桩计时）。

外加一个"API 单项耗时"按钮：直接对当前文档量 doc.selection() 取包围盒、
doc.width/height、图层树遍历各自要多少微秒 —— 插件 20ms 轮询的成本由它决定。

用法（Krita 里）：
    工具 → 脚本 → Scripter，粘贴下面一行然后运行：

    exec(open(r"E:/1/krita-ai-diffusion-main/tools/krita_perf_probe.py", encoding="utf-8").read())

    弹出小窗 → 点"开始" → 去做那些卡的操作（移动、旋转、缩放）10~30 秒
    → 点"报告"。报告同时写到 %TEMP%\\krita_perf_probe_*.log。

    A/B：勾上"暂停插件轮询"再做一遍同样的操作，两份报告对比。
    这是判定"是不是 AI 插件拖的"的唯一硬证据。

删除条件：归因结论确定并落到修复后，本文件可删；它不属于插件发布内容
（不在 ai_diffusion/ 目录下，不会被打包）。
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime

from PyQt5.QtCore import QObject, Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from krita import Krita
except ImportError:  # 允许在 Krita 外做语法检查
    Krita = None  # type: ignore

TICK_MS = 4  # 心跳间隔；小于它的卡顿测不到，也没人感觉得到
STACK_THRESHOLD_S = 0.030  # 主线程静默超过这么久就抓栈
SAMPLE_INTERVAL_S = 0.005

BUCKETS = [5, 10, 20, 50, 100, 250, 500, 1000]


def _fmt_bucket(i: int) -> str:
    if i == 0:
        return f"<{BUCKETS[0]}ms"
    if i == len(BUCKETS):
        return f">={BUCKETS[-1]}ms"
    return f"{BUCKETS[i - 1]}-{BUCKETS[i]}ms"


class _Instrument:
    """给一个函数打上计时桩，记录次数/总耗时/最大单次。"""

    def __init__(self, name: str, owner, attr: str):
        self.name = name
        self.owner = owner
        self.attr = attr
        self.original = getattr(owner, attr)
        self.calls = 0
        self.total = 0.0
        self.worst = 0.0

        original = self.original
        inst = self

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                inst.calls += 1
                inst.total += dt
                inst.worst = max(inst.worst, dt)

        setattr(owner, attr, wrapper)

    def restore(self):
        setattr(self.owner, self.attr, self.original)


# 定时器 parent 的类名 -> 它连的槽名。QTimer.timeout 在 connect 的那一刻就把
# bound method 收下了，之后替换类属性对已连接的槽毫无影响 —— 第一版报告里
# [3] 整段"未被调用"就是这么来的。所以按这张表断开重连到包装后的槽。
_TIMER_SLOTS = {
    "KritaDocument": "_poll",
    "LayerManager": "update",
    "PoseLayers": "update",
    "Model": "_poll_nai_focus_box",
}


class _TimerStub:
    """把一个定时器的槽换成计时包装。"""

    def __init__(self, label: str, timer: QTimer, owner, attr: str):
        self.label = label
        self.timer = timer
        self.calls = 0
        self.total = 0.0
        self.worst = 0.0
        self.original = getattr(owner, attr)  # bound method 或模块级函数

        original = self.original
        stub = self

        def wrapper():
            t0 = time.perf_counter()
            try:
                return original()
            finally:
                dt = time.perf_counter() - t0
                stub.calls += 1
                stub.total += dt
                stub.worst = max(stub.worst, dt)

        self.wrapper = wrapper
        timer.timeout.disconnect()
        timer.timeout.connect(wrapper)

    def restore(self):
        try:
            self.timer.timeout.disconnect(self.wrapper)
            self.timer.timeout.connect(self.original)
        except (RuntimeError, TypeError):
            pass


def _instrument_timers() -> list[_TimerStub]:
    """给当前活着的插件定时器打桩。已暂停的定时器不在其列。"""
    stubs: list[_TimerStub] = []
    for label, timer in _plugin_timers():
        try:
            parent = timer.parent()
            if parent is None:
                module = sys.modules.get("ai_diffusion.eventloop")
                if module is None:
                    continue
                stubs.append(_TimerStub(label, timer, module, "process_python_events"))
                continue
            attr = _TIMER_SLOTS.get(type(parent).__name__)
            if attr is None or not hasattr(parent, attr):
                continue
            stubs.append(_TimerStub(label, timer, parent, attr))
        except Exception:
            pass
    return stubs


def _instrument_event_filters() -> list[_Instrument]:
    """给所有 pykrita 插件的 eventFilter 打桩（含本插件之外的第三方插件）。

    全局事件过滤器在每一个 Qt 事件上都要跑一遍 Python；拖画布时每秒上千个
    鼠标事件全都要过它 —— 这是"移动旋转特别卡"的经典来源。第一版报告的
    卡顿栈里就抓到了 shortcut_composer 的 release_key_event_filter。

    eventFilter 是虚函数，每次调用都会重新查属性，所以替换类属性是生效的
    （不像信号，见 _TimerStub 的说明）。
    """
    out: list[_Instrument] = []
    seen = set()
    for name, module in list(sys.modules.items()):
        if module is None or name.startswith(("PyQt5", "krita", "sip")):
            continue
        path = str(getattr(module, "__file__", "") or "").replace("\\", "/")
        if "pykrita" not in path:
            continue
        root = name.split(".")[0]
        try:
            members = list(vars(module).items())
        except Exception:
            continue
        for _, obj in members:
            if not isinstance(obj, type) or "eventFilter" not in vars(obj):
                continue
            key = (getattr(obj, "__module__", ""), getattr(obj, "__qualname__", ""))
            if key in seen:
                continue
            seen.add(key)
            try:
                out.append(
                    _Instrument(f"{root}: {obj.__qualname__}.eventFilter", obj, "eventFilter")
                )
            except Exception:
                pass
    return out


def _plugin_timers() -> list[tuple[str, QTimer]]:
    """找出所有属于 ai_diffusion 的 QTimer。

    gc 扫描而不是写死一张表：插件里的定时器分散在 document/layer/model/ui 里，
    写死的表一改代码就过期，扫描不会。
    """
    found: list[tuple[str, QTimer]] = []
    seen = set()
    for obj in gc.get_objects():
        if not isinstance(obj, QTimer):
            continue
        try:
            if not obj.isActive():
                continue
            parent = obj.parent()
            owner = type(parent).__module__ if parent is not None else ""
            if not owner.startswith("ai_diffusion"):
                # eventloop 的定时器没有 parent，单独认领
                module = sys.modules.get("ai_diffusion.eventloop")
                if module is None or obj is not getattr(module, "_timer", None):
                    continue
                owner = "ai_diffusion.eventloop"
            key = id(obj)
            if key in seen:
                continue
            seen.add(key)
            label = f"{owner}.{type(parent).__name__ if parent else '_timer'}({obj.interval()}ms)"
            found.append((label, obj))
        except RuntimeError:
            pass  # C++ 对象已析构
    return found


def _describe_document() -> list[str]:
    lines: list[str] = []
    if Krita is None:
        return ["(不在 Krita 里运行)"]
    app = Krita.instance()
    lines.append(f"Krita 版本: {app.version()}")
    doc = app.activeDocument()
    if doc is None:
        lines.append("当前没有打开的文档")
        return lines
    lines.append(f"文档: {doc.width()}x{doc.height()} @ {doc.resolution()}dpi")
    lines.append(f"色彩: {doc.colorModel()} / {doc.colorDepth()} / {doc.colorProfile()}")
    try:
        lines.append(f"动画帧范围: {doc.fullClipRangeStartTime()}~{doc.fullClipRangeEndTime()}")
    except Exception:
        pass

    kinds: Counter[str] = Counter()
    total = 0

    def walk(node, depth=0):
        nonlocal total
        for child in node.childNodes():
            total += 1
            try:
                kinds[child.type()] += 1
                if child.type() == "grouplayer" and child.passThroughMode():
                    kinds["grouplayer(pass-through)"] += 1
            except Exception:
                pass
            walk(child, depth + 1)

    try:
        walk(doc.rootNode())
    except Exception:
        pass
    lines.append(f"图层总数: {total}")
    for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {kind}: {count}")
    sel = doc.selection()
    lines.append(f"有全局选区: {sel is not None}")
    return lines


def _read_kritarc() -> list[str]:
    """把和卡顿相关的配置逐条读出来，附上人话解释。"""
    path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "kritarc")
    interesting = {
        "OpenGLFilterMode": "画布缩放滤镜 0=最近邻 1=双线性 2=三线性 3=高质量(最贵)",
        "canvasState": "画布加速状态",
        "levelOfDetailEnabled": "即时预览(Instant Preview)总开关 —— 大画布变换/绘制的快速通道",
        "fpsLimit": "画布刷新上限，越高重绘越频繁",
        "memoryHardLimitPercent": "内存分配上限%",
        "memorySoftLimitPercent": "撤销数据上限%",
        "memoryPoolLimitPercent": "内部内存池%",
        "maxSwapSize": "交换文件上限 MiB",
        "swaplocation": "交换文件位置(在系统盘会和 OS 抢 IO)",
        "calculateAnimationCacheInBackground": "后台预渲染动画缓存",
        "animationCacheFrameSizeLimit": "动画缓存帧尺寸上限",
        "frameRenderingClones": "动画渲染线程数",
        "enableProgressReporting": "状态栏进度条(官方标注会影响性能)",
        "enablePerfLog": "Krita 自带性能日志",
        "disableAVXOptimizations": "禁用 AVX",
        "assistantsDrawMode": "绘画辅助尺渲染模式",
        "numberOfOnionSkins": "洋葱皮层数",
    }
    values: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                key, _, value = line.partition("=")
                key = key.strip()
                if key in interesting:
                    values[key] = value.strip()
    except OSError as e:
        return [f"读不到 kritarc: {e}"]
    lines = [f"kritarc: {path}"]
    for key, note in interesting.items():
        lines.append(f"    {key} = {values.get(key, '(未设置=默认)')}    # {note}")
    return lines


def _time_krita_api(iterations: int = 50) -> list[str]:
    """量插件 20ms 轮询里每个 Krita API 调用的真实成本。

    换算方法：us/次 x 50 次/秒 x 打开的文档数 = 每秒偷走的微秒数。
    """
    if Krita is None:
        return ["(不在 Krita 里运行)"]
    doc = Krita.instance().activeDocument()
    if doc is None:
        return ["当前没有打开的文档，跳过 API 计时"]

    def bench(label, fn):
        try:
            fn()  # 预热，把可缓存的东西先缓存掉
        except Exception as e:
            return f"    {label}: 失败 {e!r}"
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        dt = (time.perf_counter() - t0) / iterations
        flag = "   <<< 贵" if dt * 1000 > 0.1 else ""
        return f"    {label}: {dt * 1e6:8.1f} us/次{flag}"

    def selection_bounds():
        sel = doc.selection()
        if sel is not None:
            return (sel.x(), sel.y(), sel.width(), sel.height())
        return None

    def is_valid_check():
        # ai_diffusion.document.KritaDocument.is_valid 的原样复刻
        return doc.activeNode() is not None and doc in Krita.instance().documents()

    def walk_layers():
        count = 0
        stack = [doc.rootNode()]
        while stack:
            node = stack.pop()
            for child in node.childNodes():
                count += 1
                child.uniqueId()
                stack.append(child)
        return count

    lines = [f"Krita API 单项耗时 (每项 {iterations} 次取平均):"]
    lines.append(
        bench("is_valid = activeNode() + documents() 线性查找 [插件每 20ms 一次]", is_valid_check)
    )
    lines.append(bench("doc.selection() + 包围盒 x/y/w/h  [插件每 20ms 一次]", selection_bounds))
    lines.append(
        bench(
            "doc.width()/height()             [插件每 20ms 一次]",
            lambda: (doc.width(), doc.height()),
        )
    )
    lines.append(bench("doc.currentTime()                [插件每 20ms 一次]", doc.currentTime))
    lines.append(bench("doc.activeNode()", doc.activeNode))
    lines.append(bench("Krita.instance().documents()", Krita.instance().documents))
    lines.append(bench("图层树全遍历 + uniqueId          [插件每 500ms 一次]", walk_layers))
    return lines


class PerfProbe(QObject):
    def __init__(self):
        super().__init__()
        self.running = False
        self.timer_stubs: list[_TimerStub] = []
        self.filter_stubs: list[_Instrument] = []
        self.paused_timers: list[tuple[str, QTimer]] = []

        self._tick = QTimer(self)
        self._tick.setTimerType(Qt.TimerType.PreciseTimer)
        self._tick.setInterval(TICK_MS)
        self._tick.timeout.connect(self._on_tick)

        self._last_tick = 0.0
        self._start_time = 0.0
        self._histogram = [0] * (len(BUCKETS) + 1)
        self._worst: list[tuple[float, float]] = []  # (lag_ms, 相对开始秒数)
        self._frozen_total = 0.0  # 累计 >20ms 的卡顿时间
        self._max_lag = 0.0
        self._ticks = 0

        self._stacks: Counter[str] = Counter()
        self._sampler: threading.Thread | None = None
        self._sampler_stop = threading.Event()

    # -- 采集 --------------------------------------------------------------

    def start(self):
        if self.running:
            return
        self.running = True
        self._start_time = time.perf_counter()
        self._last_tick = self._start_time
        self._histogram = [0] * (len(BUCKETS) + 1)
        self._worst = []
        self._frozen_total = 0.0
        self._max_lag = 0.0
        self._ticks = 0
        self._stacks = Counter()

        self.timer_stubs = _instrument_timers()
        self.filter_stubs = _instrument_event_filters()
        self._tick.start()

        self._sampler_stop.clear()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._tick.stop()
        self._sampler_stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
            self._sampler = None
        for stub in self.timer_stubs:
            stub.restore()
        for inst in self.filter_stubs:
            inst.restore()

    def _on_tick(self):
        now = time.perf_counter()
        lag = (now - self._last_tick) * 1000.0 - TICK_MS
        self._last_tick = now
        self._ticks += 1
        if lag < 0:
            lag = 0.0

        index = 0
        while index < len(BUCKETS) and lag >= BUCKETS[index]:
            index += 1
        self._histogram[index] += 1

        self._max_lag = max(self._max_lag, lag)
        if lag >= 20.0:
            self._frozen_total += lag / 1000.0
            self._worst.append((lag, now - self._start_time))
            self._worst.sort(reverse=True)
            del self._worst[20:]

    def _sample_loop(self):
        """在别的线程里盯着主线程的心跳；心跳停了就抓它的 Python 栈。

        主线程调进 Krita 的 C++ 时 sip 会放开 GIL，所以这里跑得动；
        抓不到主线程帧本身就是结论 —— 卡在 C++，不是 Python。
        """
        main_id = threading.main_thread().ident
        while not self._sampler_stop.wait(SAMPLE_INTERVAL_S):
            silent = time.perf_counter() - self._last_tick
            if silent < STACK_THRESHOLD_S:
                continue
            try:
                frame = sys._current_frames().get(main_id)
            except Exception:
                continue
            if frame is None:
                self._stacks["<主线程没有 Python 栈：卡在 Krita C++ 内部>"] += 1
                continue
            stack = traceback.extract_stack(frame, limit=12)
            in_plugin = any("ai_diffusion" in f.filename.replace("\\", "/") for f in stack)
            head = " <- ".join(
                f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(stack[-6:])
            )
            tag = "[插件] " if in_plugin else "[其他 Python] "
            self._stacks[tag + head] += 1

    # -- A/B ---------------------------------------------------------------

    def pause_plugin_timers(self, pause: bool):
        if pause:
            if self.paused_timers:
                return
            self.paused_timers = _plugin_timers()
            for _, timer in self.paused_timers:
                timer.stop()
        else:
            for _, timer in self.paused_timers:
                try:
                    timer.start()
                except RuntimeError:
                    pass
            self.paused_timers = []

    # -- 报告 --------------------------------------------------------------

    def report(self) -> str:
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append(f"Krita 卡顿探针报告  {datetime.now():%Y-%m-%d %H:%M:%S}")
        lines.append(f"采样时长: {elapsed:.1f}s   心跳次数: {self._ticks}")
        if self.paused_timers:
            lines.append(f"** 本次为 A/B 对照：已暂停 {len(self.paused_timers)} 个插件定时器 **")
            for label, _ in self.paused_timers:
                lines.append(f"    暂停: {label}")
        lines.append("")

        lines.append("[1] 主线程延迟分布 (超过 4ms 心跳的部分 = 主线程被占住的时间)")
        for i, count in enumerate(self._histogram):
            if count == 0:
                continue
            share = 100.0 * count / max(self._ticks, 1)
            bar = "#" * min(int(share / 2), 40)
            lines.append(f"    {_fmt_bucket(i):>12}: {count:6d}  {share:5.1f}%  {bar}")
        lines.append(f"    最大单次卡顿: {self._max_lag:.1f} ms")
        lines.append(
            f"    累计卡住(>=20ms 的部分): {self._frozen_total:.2f}s"
            f"  = 采样时长的 {100.0 * self._frozen_total / elapsed:.1f}%"
        )
        if self._worst:
            lines.append("    最严重的几次 (卡顿 ms @ 第几秒):")
            for lag, at in self._worst[:10]:
                lines.append(f"        {lag:8.1f} ms  @ {at:6.1f}s")
        lines.append("")

        lines.append("[2] 卡住的瞬间主线程在跑什么 (按命中次数)")
        if not self._stacks:
            lines.append("    没抓到样本 —— 说明没有超过 30ms 的卡顿，或采样线程拿不到 GIL")
        for stack, count in self._stacks.most_common(12):
            lines.append(f"    {count:5d}x  {stack}")
        lines.append("")

        lines.append("[3] 插件回调的实测开销 (定时器槽 + 所有 pykrita 插件的事件过滤器)")
        if not self.timer_stubs and not self.filter_stubs:
            lines.append("    没打上桩（插件没加载，或轮询已被暂停）")
        for stub in self.timer_stubs:
            share = 100.0 * stub.total / elapsed
            lines.append(
                f"    [定时器] {stub.label}: {stub.calls} 次, 共 {stub.total * 1000:.1f}ms"
                f" ({share:.2f}% 墙钟), 最大单次 {stub.worst * 1000:.2f}ms"
            )
        hot = sorted(self.filter_stubs, key=lambda s: -s.total)
        for inst in hot[:8]:
            if inst.calls == 0:
                continue
            share = 100.0 * inst.total / elapsed
            lines.append(
                f"    [事件过滤] {inst.name}: {inst.calls} 次, 共 {inst.total * 1000:.1f}ms"
                f" ({share:.2f}% 墙钟), 最大单次 {inst.worst * 1000:.2f}ms"
            )
        idle = [i.name for i in self.filter_stubs if i.calls == 0]
        if idle:
            lines.append(f"    未触发的事件过滤器: {len(idle)} 个")
        lines.append("")

        lines.append("[4] " + "\n    ".join(_time_krita_api()))
        lines.append("")
        lines.append("[5] 当前文档")
        lines.extend("    " + line for line in _describe_document())
        lines.append("")
        lines.append("[6] 相关配置")
        lines.extend("    " + line for line in _read_kritarc())
        lines.append("")

        active = _plugin_timers()
        lines.append(f"[7] 当前活动的插件定时器: {len(active)} 个")
        for label, timer in active:
            lines.append(f"    {label} interval={timer.interval()}ms")
        lines.append("=" * 72)
        return "\n".join(lines)


class ProbeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.probe = PerfProbe()
        self.setWindowTitle("Krita 卡顿探针")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(900, 620)

        self._start_button = QPushButton("开始", self)
        self._start_button.clicked.connect(self._toggle)
        self._report_button = QPushButton("出报告", self)
        self._report_button.clicked.connect(self._report)
        self._pause_box = QCheckBox("暂停插件轮询 (A/B 对照)", self)
        self._pause_box.toggled.connect(self.probe.pause_plugin_timers)
        self._status = QLabel("未开始", self)

        self._output = QPlainTextEdit(self)
        self._output.setReadOnly(True)
        self._output.setStyleSheet("font-family: Consolas, monospace;")
        self._output.setPlainText(
            "点【开始】→ 去画布上做那些卡的操作（移动、旋转、缩放）10~30 秒 → 点【出报告】。\n"
            "然后勾上【暂停插件轮询】，把同样的操作再做一遍，出第二份报告。\n"
            "两份报告的 [1] 延迟分布如果没差别，就说明卡顿与本插件无关。"
        )

        top = QHBoxLayout()
        top.addWidget(self._start_button)
        top.addWidget(self._report_button)
        top.addWidget(self._pause_box)
        top.addWidget(self._status, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._output, 1)

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(500)
        self._ui_timer.timeout.connect(self._refresh_status)
        self._ui_timer.start()

    def closeEvent(self, event):
        self.probe.stop()  # 关窗必须还原所有打过的桩
        super().closeEvent(event)

    def _toggle(self):
        if self.probe.running:
            self.probe.stop()
            self._start_button.setText("开始")
        else:
            self.probe.start()
            self._start_button.setText("停止")

    def _refresh_status(self):
        if not self.probe.running:
            return
        elapsed = time.perf_counter() - self.probe._start_time
        self._status.setText(
            f"采集中 {elapsed:.0f}s   最大卡顿 {self.probe._max_lag:.0f}ms"
            f"   累计卡住 {self.probe._frozen_total:.1f}s"
        )

    def _report(self):
        text = self.probe.report()
        self._output.setPlainText(text)
        path = os.path.join(
            os.environ.get("TEMP", "."), f"krita_perf_probe_{datetime.now():%Y%m%d-%H%M%S}.log"
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._status.setText(f"已写入 {path}")
        except OSError as e:
            self._status.setText(f"写文件失败: {e}")
        print(text)


def main():
    app = QApplication.instance()
    assert app is not None, "需要在 Krita 里运行"
    window = ProbeWindow()
    window.show()
    # Scripter 跑完会丢掉局部变量，挂到 Krita 实例上防止被 GC 回收
    if Krita is not None:
        Krita.instance()._perf_probe_window = window
    return window


if QApplication.instance() is not None:
    _window = main()
