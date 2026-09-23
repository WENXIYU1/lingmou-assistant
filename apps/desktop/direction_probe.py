"""Seven-target descriptive probe. Never fits or validates a screen mapping."""
from dataclasses import replace
import math
import numpy as np
from .vertical_probe import VerticalProbe

TARGETS=((.5,.5),(.2,.5),(.8,.5),(.5,.5),(.5,.2),(.5,.8),(.5,.5))
NAMES=('中心','左','右','中心复测1','上','下','中心复测2')
DRIFT_WINDOW=.30

class DirectionProbe(VerticalProbe):
    @property
    def done(self):
        return len(self.groups)==len(TARGETS)

    @property
    def target(self):
        return None if self.done else TARGETS[len(self.groups)]

    def feed(self,frame):
        # Parent keeps only scalars: channel0=vertical, channel1=horizontal.
        return super().feed(replace(frame,left_eye=(0.,frame.features[0]) if frame.usable() else ()))

    def status(self):
        if self.done:
            return '方向短测完成，请查看报告；不代表校准通过。'
        return self.error or f'方向短测 {len(self.groups)+1}/7：注视{NAMES[len(self.groups)]}；准备1秒、采样2秒'

    def report(self):
        lines=['MobileGaze方向短测：仅会话内存，不保存、不上传。',
               f'完成{len(self.groups)}/7；'+self.error,
               '原始模型角度；不拟合、不纠偏、不判定定位通过；非真人延迟测试。']
        centers=[]
        for name,group in zip(NAMES,self.groups):
            xy=np.array([(v[1],v[0]) for _,v in group])*180.
            center=np.median(xy,axis=0)
            centers.append(center)
            spread=np.percentile(xy,90,axis=0)-np.percentile(xy,10,axis=0)
            lines.append(f'{name}：{len(group)}帧；角度X/Y中位{center[0]:+.2f}/{center[1]:+.2f}°；P90-P10 {spread[0]:.2f}/{spread[1]:.2f}°')
        if self.done:
            for i in (3,6):
                delta=centers[i]-centers[0]
                lines.append(f'{NAMES[i]}减首次中心：X/Y {delta[0]:+.2f}/{delta[1]:+.2f}°')
            lines.append(f'右减左X：{centers[2][0]-centers[1][0]:+.2f}°；下减上Y：{centers[5][1]-centers[4][1]:+.2f}°')
        return '\n'.join(lines)


class DriftProbe(DirectionProbe):
    """Alternating centers; scalar-only descriptive data, no correction."""
    targets=((.5,.5),(.2,.5),(.5,.5),(.8,.5),(.5,.5),(.5,.2),(.5,.5),(.5,.8),(.5,.5))
    names=('中心','左','中心复测1','右','中心复测2','上','中心复测3','下','中心复测4')

    @property
    def done(self):
        return len(self.groups)==len(self.targets)

    @property
    def target(self):
        return None if self.done else self.targets[len(self.groups)]

    def sample_values(self, frame):
        crop=frame.face_crop
        if len(crop)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in crop):
            crop=(None,)*4
        shadow=frame.shadow_features
        if len(shadow)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in shadow):
            shadow=(None,)*2
        return super().sample_values(frame)+list(frame.head)+list(crop)+list(shadow)

    def status(self):
        if self.done:
            return '九点漂移对照完成，请查看报告；不代表定位通过。'
        return self.error or f'漂移对照 {len(self.groups)+1}/9：注视{self.names[len(self.groups)]}；准备1秒、至少24帧/2秒'

    @staticmethod
    def paired(group, indices=(1,0)):
        raw, filtered, spans = [], [], []
        for i,(stamp,values) in enumerate(group):
            window=[v for t,v in group[:i+1] if 0<=stamp-t<=DRIFT_WINDOW]
            if len(window)<3:
                continue
            if any(values[j] is None for j in indices) or any(any(v[j] is None for j in indices) for v in window):
                continue
            raw.append(tuple(values[j] for j in indices))
            filtered.append(np.median([tuple(v[j] for j in indices) for v in window],axis=0))
            spans.append(stamp-group[i-len(window)+1][0])
        return np.asarray(raw)*180, np.asarray(filtered)*180, spans

    def report(self):
        lines=['MobileGaze九点漂移对照：仅会话内存，不保存、不上传。',
               f'完成{len(self.groups)}/9；'+self.error,
               '中—左—中—右—中—上—中—下—中；不拟合、不纠偏、不判定定位通过。',
               '每点独立预热；过去300毫秒至少3帧中位数，原始与处理后按相同时间点配对。',
               '头部是几何代理而非真实姿态角：中心X/Y、尺度、倾斜弧度、侧转代理。',
               '裁剪框为实际模型输入框：中心X/Y、宽/高，均除以图像宽/高；非镜像坐标。',
               '数值共同变化不能单独证明原因；窗口跨度不等于真人响应延迟。']
        medians={}
        spreads={}
        for i,(name,group) in enumerate(zip(self.names,self.groups)):
            raw,filtered,spans=self.paired(group)
            lines.append(f'\n【{name}】有效采样{len(group)}帧；配对{len(raw)}帧')
            for label,values in [('原始配对',raw),('因果中位数',filtered)]:
                if len(values)==0:
                    lines.append(label+'：配对不足'); continue
                mid=np.median(values,axis=0)
                spread=np.percentile(values,90,axis=0)-np.percentile(values,10,axis=0)
                lines.append(f'{label}角度X/Y：中位{mid[0]:+.2f}/{mid[1]:+.2f}°；P90-P10 {spread[0]:.2f}/{spread[1]:.2f}°')
                medians[i,label]=mid
                spreads[i,label]=spread
                if i in (2,4,6,8) and (0,label) in medians:
                    delta=mid-medians[0,label]
                    lines.append(f'  减首次中心：{delta[0]:+.2f}/{delta[1]:+.2f}°')
            aligned_raw,aligned_filtered,_=self.paired(group,(12,13))
            for label,values in [('稳定裁剪原始',aligned_raw),('稳定裁剪因果中位数',aligned_filtered)]:
                if len(values)==0:
                    lines.append(label+'：配对不足'); continue
                mid=np.median(values,axis=0)
                spread=np.percentile(values,90,axis=0)-np.percentile(values,10,axis=0)
                lines.append(f'{label}角度X/Y：中位{mid[0]:+.2f}/{mid[1]:+.2f}°；P90-P10 {spread[0]:.2f}/{spread[1]:.2f}°')
                medians[i,label]=mid
                spreads[i,label]=spread
                if i in (2,4,6,8) and (0,label) in medians:
                    delta=mid-medians[0,label]
                    lines.append(f'  减首次中心：{delta[0]:+.2f}/{delta[1]:+.2f}°')
            if spans:
                lines.append(f'历史窗口跨度中位{np.median(spans)*1000:.0f}毫秒；不是响应延迟')
            for label,start,end in [('头部代理',3,8),('裁剪框',8,12)]:
                values=[v[start:end] for _,v in group]
                if any(len(v)!=end-start or any(x is None for x in v) for v in values):
                    lines.append(label+'：缺少数值，不推断'); continue
                mid=np.median(values,axis=0)
                spread=np.percentile(values,90,axis=0)-np.percentile(values,10,axis=0)
                medians[i,label]=mid
                lines.append(label+'中位：'+', '.join(f'{v:+.5f}' for v in mid))
                lines.append(label+' P90-P10：'+', '.join(f'{v:.5f}' for v in spread))
                if i in (2,4,6,8) and (0,label) in medians:
                    lines.append('  减首次中心：'+', '.join(f'{v:+.5f}' for v in mid-medians[0,label]))
        if self.done:
            lines.append('\n【同帧汇总；只描述，不自动选择】')
            for label in ('原始配对','因果中位数','稳定裁剪原始','稳定裁剪因果中位数'):
                keys=[(i,label) for i in range(9)]
                if not all(key in medians for key in keys):
                    lines.append(label+'：数据不足'); continue
                horizontal=medians[3,label][0]-medians[1,label][0]
                vertical=medians[7,label][1]-medians[5,label][1]
                center_delta=np.abs(np.asarray([medians[i,label]-medians[0,label] for i in (2,4,6,8)]))
                typical=np.median(np.asarray([spreads[i,label] for i in range(9)]),axis=0)
                lines.append(f'{label}：右减左X {horizontal:+.2f}°；下减上Y {vertical:+.2f}°；'
                             f'中心复测最大绝对偏移X/Y {center_delta[:,0].max():.2f}/{center_delta[:,1].max():.2f}°；'
                             f'各点P90-P10中位X/Y {typical[0]:.2f}/{typical[1]:.2f}°')
        lines.append('\n波动减小不代表定位更准；本报告不自动选择算法或改变控制。')
        return '\n'.join(lines)


class HeadCompProbe(DriftProbe):
    """Center-gaze head-leak fit followed by untouched direction evaluation."""
    train_names=('舒适中心','头稍向本人左侧平移','头稍向本人右侧平移',
                 '头稍向上平移','头稍向下平移')
    eval_names=('中心','左','中心复测1','右','中心复测2','上','中心复测3','下','中心复测4')
    eval_targets=DriftProbe.targets
    scale=.04
    ridge=.15
    min_span=.012

    def __init__(self,reference):
        super().__init__(reference)
        self.compensation=None
        self.fit_error=''
        self.blocked=False

    @property
    def done(self):
        return self.blocked or len(self.groups)==5+len(self.eval_targets)

    @property
    def target(self):
        if self.done:
            return None
        return (.5,.5) if len(self.groups)<5 else self.eval_targets[len(self.groups)-5]

    def status(self):
        if self.blocked:
            return self.fit_error+'；已停止，请查看报告后重新开始'
        if self.done:
            return '头动补偿对照完成，请查看报告；不代表定位通过。'
        if len(self.groups)<5:
            name=self.train_names[len(self.groups)]
            return (f'头动训练 {len(self.groups)+1}/5：始终注视中心圆点；{name}。'
                    '只需舒适范围内轻微平移，不要转头。')
        name=self.eval_names[len(self.groups)-5]
        return f'独立方向评估 {len(self.groups)-4}/9：注视{name}；准备1秒、至少24帧/2秒'

    def feed(self,frame):
        changed=super().feed(frame)
        if changed and len(self.groups)==5:
            self.compensation,self.fit_error=self._fit()
            self.blocked=self.compensation is None
        return changed

    def _fit(self):
        if len(self.groups)<5:
            return None,'头动训练未完成'
        heads=[]; angles=[]
        for group in self.groups[:5]:
            heads.append(np.median([[v[3],v[4]] for _,v in group],axis=0))
            angles.append(np.median([[v[1],v[0]] for _,v in group],axis=0)*180.)
        heads=np.asarray(heads); angles=np.asarray(angles)
        spans=np.ptp(heads,axis=0)
        if spans[0]<self.min_span or spans[1]<self.min_span:
            return None,(f'头位覆盖不足：X/Y跨度{spans[0]:.5f}/{spans[1]:.5f}，'
                         f'均需≥{self.min_span:.3f}；不生成补偿')
        delta=(heads-np.asarray(self.reference[:2]))/self.scale
        design=np.column_stack((np.ones(5),delta))
        penalty=np.diag((0.,self.ridge,self.ridge))
        coefficients=np.linalg.solve(design.T@design+penalty,design.T@angles)
        if not np.isfinite(coefficients).all():
            return None,'补偿系数不是有限数；不生成补偿'
        fitted=design@coefficients
        residual=np.sqrt(np.mean((fitted-angles)**2,axis=0))
        return {'coefficients':coefficients,'heads':heads,'angles':angles,
                'spans':spans,'training_rms':residual},''

    def corrected(self,values,model):
        raw=np.asarray((values[1],values[0]),dtype=float)*180.
        delta=(np.asarray((values[3],values[4]))-np.asarray(self.reference[:2]))/self.scale
        return raw-delta@model['coefficients'][1:]

    def corrected_pairs(self,group,model):
        raw=[]; filtered=[]; spans=[]
        for i,(stamp,values) in enumerate(group):
            window=[v for t,v in group[:i+1] if 0<=stamp-t<=DRIFT_WINDOW]
            if len(window)<3:
                continue
            samples=[self.corrected(v,model) for v in window]
            if not all(np.isfinite(v).all() for v in samples):
                continue
            raw.append(samples[-1]); filtered.append(np.median(samples,axis=0))
            spans.append(stamp-group[i-len(window)+1][0])
        return np.asarray(raw),np.asarray(filtered),spans

    def report(self):
        model,error=(self.compensation,self.fit_error) if self.compensation is not None or self.fit_error else self._fit()
        lines=['MobileGaze头动泄漏补偿对照：仅会话内存，不保存、不上传。',
               f'完成{len(self.groups)}/14；',
               '前5组始终注视中心，仅用于拟合头部中心X/Y引起的角度串扰；后9组不参与拟合。',
               '补偿为实验影子结果，不自动采用、不拟合屏幕位置、不解锁控制。',
               '300毫秒因果中位数至少3帧；窗口跨度不等于真人响应延迟。']
        if model is None:
            lines.append('不可评估：'+error)
            return '\n'.join(lines)
        coefficients=model['coefficients']
        lines.extend([f'训练头位X/Y跨度：{model["spans"][0]:.5f}/{model["spans"][1]:.5f}',
                      '每移动画面归一化0.01时的预测角度变化X/Y：',
                      f'  头部X：{coefficients[1,0]/4:+.2f}/{coefficients[1,1]/4:+.2f}°',
                      f'  头部Y：{coefficients[2,0]/4:+.2f}/{coefficients[2,1]/4:+.2f}°',
                      f'训练内拟合RMS X/Y：{model["training_rms"][0]:.2f}/{model["training_rms"][1]:.2f}°（非独立证据）'])
        for name,head,angle in zip(self.train_names,model['heads'],model['angles']):
            lines.append(f'{name}：头部X/Y {head[0]:+.5f}/{head[1]:+.5f}；原始角度X/Y {angle[0]:+.2f}/{angle[1]:+.2f}°')
        metrics={}; spreads={}
        for i,(name,group) in enumerate(zip(self.eval_names,self.groups[5:])):
            raw,base_filtered,spans=self.paired(group)
            corrected,corrected_filtered,_=self.corrected_pairs(group,model)
            lines.append(f'\n【独立评估·{name}】有效采样{len(group)}帧；配对{len(raw)}帧')
            for label,values in [('原始',raw),('原始因果中位数',base_filtered),
                                 ('头动补偿',corrected),('头动补偿因果中位数',corrected_filtered)]:
                if len(values)==0:
                    lines.append(label+'：数据不足'); continue
                mid=np.median(values,axis=0)
                spread=np.percentile(values,90,axis=0)-np.percentile(values,10,axis=0)
                metrics[i,label]=mid; spreads[i,label]=spread
                lines.append(f'{label}角度X/Y：中位{mid[0]:+.2f}/{mid[1]:+.2f}°；P90-P10 {spread[0]:.2f}/{spread[1]:.2f}°')
                if i in (2,4,6,8) and (0,label) in metrics:
                    delta=mid-metrics[0,label]
                    lines.append(f'  减首次中心：{delta[0]:+.2f}/{delta[1]:+.2f}°')
            if spans:
                lines.append(f'历史窗口跨度中位{np.median(spans)*1000:.0f}毫秒；不是响应延迟')
        if len(self.groups)>=14:
            lines.append('\n【独立九点汇总；训练组不参与】')
            for label in ('原始','原始因果中位数','头动补偿','头动补偿因果中位数'):
                if not all((i,label) in metrics for i in range(9)):
                    lines.append(label+'：数据不足'); continue
                horizontal=metrics[3,label][0]-metrics[1,label][0]
                vertical=metrics[7,label][1]-metrics[5,label][1]
                centers=np.abs(np.asarray([metrics[i,label]-metrics[0,label] for i in (2,4,6,8)]))
                typical=np.median([spreads[i,label] for i in range(9)],axis=0)
                lines.append(f'{label}：右减左X {horizontal:+.2f}°；下减上Y {vertical:+.2f}°；'
                             f'中心复测最大绝对偏移X/Y {centers[:,0].max():.2f}/{centers[:,1].max():.2f}°；'
                             f'各点P90-P10中位X/Y {typical[0]:.2f}/{typical[1]:.2f}°')
        lines.append('\n训练内RMS不能证明补偿有效；只比较后9组。结果不会自动改变算法。')
        return '\n'.join(lines)
