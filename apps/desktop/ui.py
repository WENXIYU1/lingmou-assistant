"""CustomTkinter practice UI; synthetic input, no camera or OS mouse output."""
from dataclasses import replace
import time
import tkinter as tk
from tkinter import messagebox
import uuid
import customtkinter as ctk
from .calibration import AdaptationStatus
from .interaction import Controller
from .executor import PracticeAction, PracticeExecutor
from .storage import default_path, load_settings, save_settings
from .vision import Observation


class PracticeApp(ctk.CTk):
    def __init__(self, settings_path=None):
        super().__init__()
        self.title('灵动视眸 · 第一步安全练习（模拟输入）')
        self.geometry('1080x760')
        self.minsize(960, 680)
        self.settings_path = settings_path or default_path()
        self.controller = Controller()
        self.load_error = ''
        try:
            self.controller.configure(load_settings(self.settings_path))
        except (ValueError, OSError, TypeError) as exc:
            self.load_error = f'配置未加载：{exc}。当前使用默认练习参数；原文件保留。'
        self.executor = PracticeExecutor(self.controller)
        self.adaptation = AdaptationStatus()
        self.left = self.right = self.mouth = False
        self.raw = (0.5, 0.5)
        self.signal = True
        self.actions = 0
        self.target_hits = 0
        self.target_index = 0
        self.targets = [(0.25, 0.3), (0.75, 0.3), (0.75, 0.7), (0.25, 0.7)]
        self.status = tk.StringVar(value='已暂停')
        self.result = tk.StringVar(value='模拟确认 0 次；大目标命中 0 次（非眼控准确率）')
        self.entries = {}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(self, text='灵动视眸 / 安全练习', font=('Microsoft YaHei', 24, 'bold')).grid(
            row=0, column=0, sticky='w', padx=24, pady=(18, 4))
        ctk.CTkLabel(self, text='仅模拟：不会开启摄像头、移动系统鼠标或执行桌面点击。',
                     font=('Microsoft YaHei', 16)).grid(row=1, column=0, padx=24, sticky='w')
        self.canvas = tk.Canvas(self, bg='#122335', highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky='nsew', padx=24, pady=14)
        self.canvas.bind('<Motion>', self.on_motion)
        self.canvas.bind('<Configure>', lambda event: self.redraw())
        side = ctk.CTkScrollableFrame(self, width=300)
        side.grid(row=0, column=1, rowspan=5, sticky='nsew', padx=(0, 20), pady=18)
        ctk.CTkLabel(side, text='未校准 · 系统控制锁定', font=('Microsoft YaHei', 18, 'bold')).pack(pady=10)
        ctk.CTkButton(side, text='首次适配说明', height=44, command=self.show_adaptation).pack(fill='x', pady=5)
        ctk.CTkButton(side, text='开始 / 明确恢复练习', height=44, command=self.resume).pack(fill='x', pady=5)
        ctk.CTkButton(side, text='立即暂停（Esc）', height=48, fg_color='#9d3434', command=self.pause).pack(fill='x', pady=5)
        ctk.CTkButton(side, text='重置中心并暂停', height=40, command=self.reset).pack(fill='x', pady=5)
        for key, label in [('gain_x', '横向增益'), ('gain_y', '纵向增益'),
                           ('deadzone_x', '横向死区'), ('deadzone_y', '纵向死区'),
                           ('smoothing_seconds', '平滑时间常数 / 秒'), ('hold_seconds', '持续动作时间 / 秒')]:
            ctk.CTkLabel(side, text=label).pack(anchor='w')
            entry = ctk.CTkEntry(side, height=32)
            entry.insert(0, str(getattr(self.controller.settings, key)))
            entry.pack(fill='x', pady=(0, 4))
            self.entries[key] = entry
        ctk.CTkButton(side, text='应用练习参数（会暂停）', height=40, command=self.apply_settings).pack(fill='x', pady=5)
        ctk.CTkButton(side, text='保存练习参数（不是校准）', height=40, command=self.save).pack(fill='x', pady=5)
        controls = ctk.CTkFrame(self)
        controls.grid(row=3, column=0, sticky='ew', padx=24)
        for label, action in [('模拟短眨', self.short_blink), ('模拟持续左眼动作', self.long_wink),
                              ('模拟丢失', self.lose_signal), ('恢复信号（不启用）', self.restore_signal)]:
            ctk.CTkButton(controls, text=label, width=150, height=42, command=action).pack(side='left', padx=3, pady=8)
        footer = ctk.CTkFrame(self, fg_color='transparent')
        footer.grid(row=4, column=0, sticky='ew', padx=24, pady=12)
        ctk.CTkLabel(footer, textvariable=self.status, wraplength=680).pack(anchor='w')
        ctk.CTkLabel(footer, textvariable=self.result).pack(anchor='w')
        ctk.CTkLabel(footer, text='画布内移动鼠标模拟特征；离开仅保持位置。用“模拟丢失”测试暂停。参数未经真人验证。',
                     wraplength=680).pack(anchor='w')
        self.bind('<Escape>', lambda event: self.pause())
        self.protocol('WM_DELETE_WINDOW', self.shutdown)
        self.after(40, self.tick)

    def on_motion(self, event):
        self.raw = (min(1, max(0, event.x / max(1, self.canvas.winfo_width()))),
                    min(1, max(0, event.y / max(1, self.canvas.winfo_height()))))

    def resume(self):
        self.controller.resume()

    def pause(self):
        self.controller.pause()
        self.redraw()

    def reset(self):
        self.left = self.right = self.mouth = False
        self.pause()

    def lose_signal(self):
        self.signal = False
        self.controller.lose_tracking()

    def restore_signal(self):
        self.signal = True

    def short_blink(self):
        self.left = self.right = True
        self.after(120, self.release_eyes)

    def long_wink(self):
        self.left, self.right = True, False
        self.after(int((self.controller.settings.hold_seconds + 0.3) * 1000), self.release_eyes)

    def release_eyes(self):
        self.left = self.right = False

    def show_adaptation(self):
        self.pause()
        messagebox.showinfo('首次适配：当前为前置入口', self.adaptation.message +
                            '\n下一阶段：位置检查 → 五点校准 → 独立验证 → 灵敏度试用 → 个人档案。'
                            '\n本页面不会生成“已校准”标记。', parent=self)

    def apply_settings(self):
        self.pause()
        try:
            settings = replace(self.controller.settings,
                               **{key: float(entry.get()) for key, entry in self.entries.items()})
            self.controller.configure(settings)
            return True
        except ValueError as exc:
            messagebox.showerror('参数未应用', str(exc), parent=self)
            return False

    def save(self):
        if self.apply_settings():
            try:
                save_settings(self.settings_path, self.controller.settings)
                messagebox.showinfo('已保存', '只保存练习参数，不代表个人校准通过。', parent=self)
            except (ValueError, OSError, TypeError) as exc:
                messagebox.showerror('保存失败', str(exc), parent=self)

    def redraw(self):
        self.canvas.delete('all')
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        tx, ty = self.targets[self.target_index]
        self.canvas.create_rectangle((tx - .10) * w, (ty - .13) * h,
                                     (tx + .10) * w, (ty + .13) * h,
                                     fill='#244961', outline='#91b9d4', width=3)
        self.canvas.create_text(tx * w, ty * h, text='练习目标', fill='white', font=('Microsoft YaHei', 18))
        x, y = self.controller.motion.position
        self.canvas.create_oval(x*w-10, y*h-10, x*w+10, y*h+10, fill='#ffd77a', outline='white')
        self.canvas.create_text(16, 18, anchor='nw', text='模拟光标 · 非真实眼动', fill='white')

    def tick(self):
        if self.controller.closed:
            return
        now = time.monotonic()
        self.controller.watchdog(now)
        result = self.controller.observe(Observation(now, *self.raw, valid=self.signal,
                                           left_closed=self.left, right_closed=self.right,
                                           mouth_open=self.mouth))
        if result and result[1]:
            action = PracticeAction(str(uuid.uuid4()), self.controller.epoch, result[1])
            if self.executor.execute(action):
                self.actions += 1
                x, y = result[0]
                tx, ty = self.targets[self.target_index]
                if abs(x-tx) <= .10 and abs(y-ty) <= .13:
                    self.target_hits += 1
                    self.target_index = (self.target_index + 1) % len(self.targets)
                self.result.set(f'模拟确认 {self.actions} 次；大目标命中 {self.target_hits} 次（非眼控准确率）')
        self.status.set(self.load_error or ('练习中 · 系统操作锁定' if self.controller.can_practice
                                            else '已暂停 · 信号恢复后仍需明确恢复'))
        self.redraw()
        self.after(40, self.tick)

    def shutdown(self):
        self.controller.close()
        self.destroy()
