"""Opt-in camera, five-point wizard, held-out check, and local profile practice."""
from dataclasses import replace
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from .camera import CameraService
from .calibration import CalibrationSession
from .config import Settings
from .eye_features import FeatureFrame, head_matches
from .interaction import MotionFilter
from .profiles import load_profile, save_profile, profile_path
from .storage import default_path
from .vision import Observation


class CameraApp(ctk.CTk):
    def __init__(self, service=None, profiles_directory=None):
        super().__init__()
        self.title('灵动视眸 · 本地摄像头适配（系统控制锁定）')
        self.geometry('1160x800')
        self.minsize(1000, 720)
        self.service = service or CameraService()
        self.profiles_directory = Path(profiles_directory or default_path().parent / 'profiles')
        self.settings = Settings()
        self.filter = self.new_filter()
        self.session = None
        self.mapping = None
        self.latest = None
        self.checked = self.active = self.closing = False
        self.bound_environment = self.bound_name = None
        self.camera_index = None
        self.camera_label = None
        self.trial_hits = 0
        self.dwell_since = None
        self.practice_targets = ((.25,.35), (.75,.35), (.5,.7))
        self.position = None
        self.message = tk.StringVar(value='摄像头未开启。先填写个人别名，阅读说明后手动开启。')
        self.health = tk.StringVar(value='摄像头关闭 / 不录制 / 不上传 / 系统鼠标锁定')
        self.name = tk.StringVar(value='用户1')
        self.name.trace_add('write', lambda *args: self.invalidate('已切换档案，请重新校准或加载本人档案'))
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(self, text='首次适配：让系统适应你', font=('Microsoft YaHei',24,'bold')).grid(
            row=0,column=0,sticky='w',padx=22,pady=(16,4))
        ctk.CTkLabel(self,textvariable=self.health,wraplength=750).grid(row=1,column=0,sticky='w',padx=22)
        self.canvas=tk.Canvas(self,bg='#122335',highlightthickness=0)
        self.canvas.grid(row=2,column=0,sticky='nsew',padx=22,pady=12)
        panel=ctk.CTkScrollableFrame(self,width=300)
        panel.grid(row=0,column=1,rowspan=5,sticky='nsew',padx=(0,18),pady=16)
        ctk.CTkLabel(panel,text='① 键鼠/照护者辅助设置\n② 双眼可用 · 注视停留练习\n单眼/扫描模式本轮尚不支持',justify='left').pack(pady=8)
        ctk.CTkLabel(panel,text='本人别名（不要填病情或身份信息）').pack(anchor='w')
        ctk.CTkEntry(panel,textvariable=self.name).pack(fill='x',pady=4)
        self.index=tk.StringVar(value='0')
        self.device_label=tk.StringVar(value='内置摄像头')
        ctk.CTkLabel(panel,text='摄像头编号 / 自定义设备标记').pack(anchor='w')
        ctk.CTkEntry(panel,textvariable=self.index).pack(fill='x',pady=3)
        ctk.CTkEntry(panel,textvariable=self.device_label).pack(fill='x',pady=3)
        for label,command in [('开启本地摄像头（需确认）',self.start_camera),
                              ('关闭摄像头',self.stop_camera),
                              ('开始新的五点校准',self.start_calibration),
                              ('继续 / 重采当前点',self.continue_point),
                              ('加载本人档案并快速验证',self.load),
                              ('明确开始 / 恢复画布试用',self.resume),
                              ('立即暂停（Esc）',self.pause),
                              ('确认保存个人档案',self.save)]:
            ctk.CTkButton(panel,text=label,height=40,command=command).pack(fill='x',pady=4)
        ctk.CTkLabel(panel,text='参数均为试验值；调整后必须重新验证').pack(pady=(10,3))
        self.entries={}
        for key,label in [('gain_x','横向增益'),('gain_y','纵向增益'),
                          ('deadzone_x','横向死区'),('deadzone_y','纵向死区'),
                          ('smoothing_seconds','平滑时间常数/秒'),('hold_seconds','停留确认时间/秒')]:
            ctk.CTkLabel(panel,text=label).pack(anchor='w')
            entry=ctk.CTkEntry(panel)
            entry.insert(0,str(getattr(self.settings,key)))
            entry.pack(fill='x',pady=2)
            self.entries[key]=entry
        ctk.CTkButton(panel,text='应用参数并重新验证',command=self.apply_settings,height=40).pack(fill='x',pady=8)
        ctk.CTkLabel(self,textvariable=self.message,wraplength=740,justify='left',font=('Microsoft YaHei',16)).grid(
            row=3,column=0,sticky='ew',padx=22,pady=8)
        ctk.CTkLabel(self,text='依次注视圆点；开始后自动采样。可随时暂停、重采或关闭。\n无需眼控点击小按钮；不保存原始画面。验证通过也只允许本窗口练习。',
                     wraplength=740,justify='left').grid(row=4,column=0,sticky='w',padx=22,pady=(0,18))
        self.bind('<Escape>',lambda event:self.pause())
        self.protocol('WM_DELETE_WINDOW',self.shutdown)
        self.after(50,self.tick)

    def new_filter(self):
        return MotionFilter(replace(self.settings,gain_x=1,gain_y=1),absolute=True)

    def environment(self):
        if self.latest is None:
            raise ValueError('尚未收到摄像头尺寸')
        return {'camera_index':self.camera_index,'device_label':self.camera_label,'backend':'DSHOW',
                'capture_size':list(self.latest.capture_size),
                'screen':[self.winfo_screenwidth(),self.winfo_screenheight(),round(float(self.tk.call('tk','scaling')),4)],
                'canvas':[self.canvas.winfo_rootx(),self.canvas.winfo_rooty(),self.canvas.winfo_width(),self.canvas.winfo_height()],
                'model':'face-landmarker-float16-1'}

    def invalidate(self,message):
        self.checked=self.active=False
        self.session=self.mapping=None
        self.bound_environment=self.bound_name=None
        self.trial_hits=0
        self.dwell_since=None
        self.position=None
        self.filter=self.new_filter()
        if hasattr(self,'message'):
            self.message.set(message)

    def fresh(self):
        return self.latest is not None and self.latest.usable() and 0 <= time.monotonic()-self.latest.timestamp <= .25

    def start_camera(self):
        if self.service.running:
            self.message.set('采集仍在运行或正在退出，请先关闭并等待')
            return
        if not messagebox.askyesno('开启摄像头', '仅在本机分析眼部特征；不保存或上传画面。\n请保持画面中只有你一人。是否开启？',parent=self):
            return
        try:
            index=int(self.index.get())
            label=self.device_label.get().strip()
            if not label or len(label)>64:
                raise ValueError('请填写1–64字的设备标记')
            self.invalidate('摄像头正在初始化；请面向摄像头，保持双眼清晰可见')
            self.latest=None
            self.camera_index,self.camera_label=index,label
            self.service.start(index)
        except (ValueError,RuntimeError,OSError) as exc:
            self.message.set(str(exc))

    def stop_camera(self):
        self.invalidate('正在关闭摄像头，等待设备释放')
        self.latest=None
        self.service.stop()

    def start_calibration(self):
        try:
            profile_path(self.profiles_directory,self.name.get())
            if not self.fresh():
                raise ValueError('请先开启摄像头，并等待双眼质量合格')
            self.invalidate('新的校准即将开始')
            self.bound_environment=self.environment()
            self.bound_name=self.name.get()
            self.session=CalibrationSession(self.settings,(self.canvas.winfo_width(),self.canvas.winfo_height()))
            self.session.begin_point(time.monotonic())
            self.message.set('校准1/5：注视圆点，保持头部稳定；短暂闭眼会重采本点稳定片段')
        except ValueError as exc:
            self.message.set(str(exc))

    def continue_point(self):
        if self.session and self.session.phase in ('train','verify') and self.fresh():
            self.session.error=''
            self.session.begin_point(time.monotonic())
            self.message.set('本点重新采样，请注视目标')
        else:
            self.message.set('需要可用摄像头和未完成的采样流程；否则请重新校准')

    def pause(self):
        self.active=False
        self.dwell_since=None
        self.position=None
        self.filter=self.new_filter()
        if self.session:
            self.session.pause()
        self.message.set('已主动暂停；不会因闭嘴或恢复跟踪而继续。可继续本点或明确恢复试用。')

    def resume(self):
        if not self.checked or not self.mapping or not self.fresh():
            self.message.set('需完成独立验证且跟踪有效，才能开始画布试用')
            return
        if self.environment()!=self.bound_environment or self.name.get()!=self.bound_name:
            self.invalidate('用户或显示配置已变化，请重新验证')
            return
        if not head_matches(self.mapping.head,self.latest.head):
            self.message.set('头部位置偏离校准范围，请调整或重新校准')
            return
        self.filter=self.new_filter()
        self.active=True
        self.dwell_since=None
        self.message.set('画布试用：依次停留在3个大目标；可暂停或调整参数。不会操作其他软件。')

    def apply_settings(self):
        try:
            new=replace(self.settings,**{k:float(e.get()) for k,e in self.entries.items()})
            self.pause()
            self.settings=new
            self.filter=self.new_filter()
            self.checked=False
            self.trial_hits=0
            if self.mapping and self.bound_environment==self.environment():
                self.session=CalibrationSession(new,(self.canvas.winfo_width(),self.canvas.winfo_height()),self.mapping)
                self.message.set('参数已改变；请点击“继续 / 重采当前点”完成3点独立验证')
            else:
                self.session=None
                self.message.set('参数已改变，请重新校准')
        except (ValueError,TypeError) as exc:
            self.message.set(f'参数未应用：{exc}')

    def load(self):
        try:
            if not self.fresh():
                raise ValueError('先开启摄像头并保持双眼可见')
            environment=self.environment()
            data,mapping,settings=load_profile(self.profiles_directory,self.name.get(),environment)
            self.invalidate('档案已加载，但尚未通过本次快速验证')
            self.settings=settings
            self.filter=self.new_filter()
            self.mapping=mapping
            self.bound_environment,self.bound_name=environment,self.name.get()
            for key,entry in self.entries.items():
                entry.delete(0,'end'); entry.insert(0,str(getattr(settings,key)))
            self.session=CalibrationSession(settings,(self.canvas.winfo_width(),self.canvas.winfo_height()),mapping)
            self.message.set('档案加载成功；请点击“继续 / 重采当前点”开始3点快速验证。系统控制仍锁定。')
        except (ValueError,OSError,TypeError,KeyError,OverflowError) as exc:
            self.invalidate(f'档案未加载，原文件保留：{exc}')

    def save(self):
        if (not self.checked or self.trial_hits<3 or not self.session or not self.session.metrics
                or not self.mapping or self.name.get()!=self.bound_name):
            self.message.set('先完成独立验证及3个画布目标，再确认保存')
            return
        self.pause()
        if not messagebox.askyesno('保存本人档案','确认当前定位和灵敏度适合本次试用？\n这不代表医学评估，也不开放系统操作。',parent=self):
            return
        try:
            if self.environment()!=self.bound_environment:
                raise ValueError('窗口或设备配置变化，请重新校准')
            save_profile(self.profiles_directory,self.name.get(),self.bound_environment,self.mapping,
                         self.settings,self.session.metrics,self.trial_hits)
            self.message.set('本人档案已保存。下次加载仍需快速验证；系统鼠标始终锁定。')
        except (ValueError,OSError,TypeError,KeyError) as exc:
            self.message.set(f'保存失败，未覆盖无效原档案：{exc}')

    def process_frame(self,frame):
        if self.latest and frame.timestamp<=self.latest.timestamp:
            self.active=False
            self.position=None
            self.dwell_since=None
            if self.session:
                self.session.buffer=[]
            return
        self.latest=frame
        if self.bound_environment and self.environment()!=self.bound_environment:
            self.invalidate('窗口、显示或摄像头尺寸已改变，旧映射不再有效，请重新校准')
        if not self.fresh():
            self.active=False
            self.position=None
            self.dwell_since=None
            if self.session:
                self.session.feed(FeatureFrame(frame.timestamp,reason='过期或无效帧'))
            return
        if self.session and self.session.phase in ('train','verify'):
            changed=self.session.feed(frame)
            if changed:
                self.mapping=self.session.mapping
                if self.session.phase in ('train','verify'):
                    self.session.begin_point(time.monotonic())
                elif self.session.phase=='trial':
                    self.checked=True
                    m=self.session.metrics
                    self.message.set(f'独立验证达到试验门槛：中位误差{m["median_error"]:.1%}、最大{m["max_error"]:.1%}画布对角线。请明确开始试用。')
                else:
                    self.message.set(self.session.error)
            if self.session.phase in ('train','verify'):
                name='五点校准' if self.session.phase=='train' else '独立验证'
                count=5 if self.session.phase=='train' else 3
                self.message.set(self.session.error or f'{name} {self.session.index+1}/{count}：有效片段{len(self.session.buffer)}帧；'+
                                 ('正在自动采样' if self.session.started is not None else '已暂停，需明确继续'))
        if self.active and self.mapping:
            if not head_matches(self.mapping.head,frame.head):
                self.pause()
                self.message.set('头部偏离校准范围，已暂停；调整姿态后明确恢复，或重新校准')
                return
            raw=self.mapping.predict(frame.features,self.settings)
            if any(not math.isfinite(v) or not -.5<=v<=1.5 for v in raw):
                self.pause()
                self.message.set('映射超出可靠范围，已暂停，请重新校准')
                return
            self.position=self.filter.update(Observation(frame.timestamp,*[min(1,max(0,v)) for v in raw]))
            target=self.practice_targets[min(self.trial_hits,2)]
            inside=all(abs(a-b)<.10 for a,b in zip(self.position,target))
            if self.trial_hits<3 and inside:
                if self.dwell_since is None:
                    self.dwell_since=frame.timestamp
                elif frame.timestamp-self.dwell_since>=self.settings.hold_seconds:
                    self.trial_hits+=1
                    self.dwell_since=None
                    self.message.set(f'试用完成{self.trial_hits}/3。'+('可以确认保存；不满意可调整参数重新验证。' if self.trial_hits==3 else '请注视下一个大目标'))
            else:
                self.dwell_since=None

    def redraw(self):
        self.canvas.delete('all')
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        target=None
        if self.session and self.session.phase in ('train','verify'):
            target=self.session.target
        elif self.checked:
            target=self.practice_targets[min(self.trial_hits,2)]
        if target:
            x,y=target[0]*w,target[1]*h
            self.canvas.create_oval(x-30,y-30,x+30,y+30,fill='#ffdc86',outline='white',width=3)
            self.canvas.create_text(x,y-50,text='注视这里',fill='white',font=('Microsoft YaHei',18))
        else:
            self.canvas.create_text(w/2,h/2,text='环境检查：面向摄像头、双眼可见、减少侧转\n仅本地特征分析，不显示或保存原始画面',fill='white',font=('Microsoft YaHei',18))
        if self.position and self.active:
            x,y=self.position[0]*w,self.position[1]*h
            self.canvas.create_oval(x-9,y-9,x+9,y+9,outline='#74ddff',width=3)

    def tick(self):
        lost,item=self.service.poll()
        if lost:
            self.active=False
            self.position=None
            self.dwell_since=None
            self.filter=self.new_filter()
            if self.checked:
                self.message.set('跟踪质量下降，试用已暂停；信号恢复后需要明确恢复')
            if self.session:
                self.session.buffer=[]
        if item:
            kind,payload=item
            if kind=='frame':
                self.process_frame(payload)
                self.health.set('摄像头采集中 / '+payload.reason+' / 不保存、不上传 / 系统控制锁定')
            else:
                self.invalidate(str(payload))
                self.latest=None
                self.health.set('摄像头已停止或故障 / 系统控制锁定')
        if self.latest and not self.fresh():
            self.active=False
            self.position=None
            self.dwell_since=None
            if self.service.running:
                self.health.set('摄像头采集中，但当前帧无效或过期 / 试用暂停 / 系统控制锁定')
            if self.session and self.session.started is not None:
                self.session.buffer=[]
        if self.bound_environment and self.latest and self.environment()!=self.bound_environment:
            self.invalidate('窗口/屏幕配置改变，请重新校准')
        if self.closing:
            if not self.service.running:
                self.destroy()
                return
        else:
            self.redraw()
        self.after(50,self.tick)

    def shutdown(self):
        self.closing=True
        self.stop_camera()
        if not self.service.running:
            self.destroy()
