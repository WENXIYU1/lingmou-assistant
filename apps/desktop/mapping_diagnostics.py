"""Session-only shadow comparisons. No selection, persistence, or control output."""
from dataclasses import replace
import math
import numpy as np

CHANNELS = (('average', '双眼平均'), ('left_eye', '左眼'), ('right_eye', '右眼'),
            ('average_lid', '眼睑纵向·双眼'), ('left_lid', '眼睑纵向·左眼'), ('right_lid', '眼睑纵向·右眼'))
TRAIN_GATE_VERSION = 'training-point-median-v1'
TRAIN_MEDIAN_LIMIT = .10


def training_quality(rows):
    if len(rows)!=5:
        return False, '训练诊断不完整'
    worst=max(rows,key=lambda row:row['median'])
    passed=all(math.isfinite(row['median']) and row['median']<=TRAIN_MEDIAN_LIMIT for row in rows)
    return passed, (f'训练质量检查 {TRAIN_GATE_VERSION}：最差点{worst["point"]}'
                    f'中位误差{worst["median"]:.2%}，实验门槛≤{TRAIN_MEDIAN_LIMIT:.0%}；'
                    + ('通过不代表独立验证通过' if passed else '不可靠，已停止进入独立验证'))


def vertical_signal(groups):
    values=[np.array([f.features[1] for f in group]) for group in groups]
    centers=[float(np.median(v)) for v in values]
    spreads=[float(np.quantile(v,.9)-np.quantile(v,.1)) for v in values]
    return {'centers':centers,'spreads':spreads,
            'down_minus_up':(centers[3]+centers[4]-centers[1]-centers[2])/2}


def channel_groups(groups, channel):
    result=[]
    for group in groups:
        converted=[]
        for frame in group:
            values=frame.features if channel=='average' else getattr(frame,channel)
            if len(values)!=2 or not all(math.isfinite(v) for v in values):
                raise ValueError('缺少有效的分眼特征（不能由双眼平均值反推）')
            converted.append(replace(frame,features=values))
        result.append(converted)
    return result


def summarize(mapping, groups, targets, settings, size):
    from .eye_features import head_matches
    if len(groups)!=len(targets) or any(len(g)<18 for g in groups):
        raise ValueError('诊断采样不完整')
    scale=np.array(size,dtype=float)
    diagonal=float(np.linalg.norm(scale))
    rows=[]
    for index,(target,group) in enumerate(zip(targets,groups),1):
        if any(not f.usable() or not head_matches(mapping.head,f.head) for f in group):
            raise ValueError('诊断帧或姿态无效')
        predictions=np.array([mapping.predict(f.features,settings) for f in group])
        errors=(predictions-target)*scale
        distances=np.linalg.norm(errors,axis=1)/diagonal
        heads=np.array([f.head for f in group])
        delta=heads-np.array(mapping.head)
        delta[:,2]=heads[:,2]/mapping.head[2]-1
        rows.append({'point':index,'median':float(np.median(distances)),
                     'max':float(max(distances)),
                     'axis_abs_px':tuple(float(v) for v in np.median(np.abs(errors),axis=0)),
                     'bias_px':tuple(float(v) for v in np.median(errors,axis=0)),
                     'head_delta':tuple(float(v) for v in np.median(delta,axis=0)),
                     'head_max_abs':tuple(float(v) for v in np.max(np.abs(delta),axis=0))})
    return rows


class MappingDiagnostics:
    def __init__(self):
        self.candidates={}
        self.training={}
        self.validation={}
        self.errors={}
        self.signals={}
        self.fit_audit={}
        self.amplification={}
        self.baseline=None

    def fit(self, groups, reference, settings, size):
        from .calibration import fit_mapping, TRAIN_TARGETS
        # Called once at training completion, before validation exists.
        for channel,label in CHANNELS:
            try:
                data=channel_groups(groups,channel)
                self.signals[channel]=vertical_signal(data)
                self.fit_audit[channel]=[]
                mapping=fit_mapping(data,reference,self.fit_audit[channel])
                # Exact affine response to +0.001 in each input feature, pixels.
                self.amplification[channel]=tuple(tuple(
                    .001*mapping.coefficients[i][j]/mapping.scale[i]*size[j]*
                    (settings.gain_x if j==0 else settings.gain_y) for j in range(2)) for i in range(2))
                self.candidates[channel]=mapping
                self.training[channel]=summarize(mapping,data,TRAIN_TARGETS,settings,size)
                if channel=='average':
                    old=fit_mapping(data,reference,stabilize=False)
                    self.baseline={'mapping':old,'training':summarize(old,data,TRAIN_TARGETS,settings,size)}
            except ValueError as exc:
                self.errors[channel]=str(exc)

    def evaluate(self, groups, settings, size):
        from .calibration import validate_mapping, CHECK_TARGETS
        self.validation={}
        if self.baseline is not None:
            try:
                self.baseline['validation']=validate_mapping(self.baseline['mapping'],groups,settings,size)
            except ValueError as exc:
                self.baseline['error']=str(exc)
        for channel,mapping in self.candidates.items():
            try:
                data=channel_groups(groups,channel)
                metrics=validate_mapping(mapping,data,settings,size)
                self.validation[channel]=(metrics,summarize(mapping,data,CHECK_TARGETS,settings,size))
            except ValueError as exc:
                self.errors[channel]=str(exc)

    def report(self):
        lines=['定位诊断（仅本次会话，不保存、不上传）',
               '候选只用五点训练；三点验证不参与拟合或自动选择。',
               '训练误差不代表独立精度；若据此选择候选，仍需新的独立验收。',
               '百分比为画布对角线比例；X/Y像素为横/纵绝对误差中位数。',
               '头部变化顺序：中心X、中心Y、尺度比例、倾斜弧度、侧转代理。',
               '这些是数值描述，不能单独证明误差原因。']
        lines.append('当前映射v2：使用训练点内波动约束放大程度；旧v1仅作诊断对照。')
        if self.baseline is not None:
            lines.append('旧v1各训练点中位误差：'+', '.join(f'{r["median"]:.2%}' for r in self.baseline['training']))
            if 'validation' in self.baseline:
                m=self.baseline['validation']
                lines.append(f'旧v1独立验证中位{m["median_error"]:.2%} / 最大{m["max_error"]:.2%}')
        for channel,label in CHANNELS:
            lines.append('\n【'+label+'】')
            for row in self.fit_audit.get(channel,[]):
                lines.append(f'拟合点{row["point"]}：输入{row["input"]} 保留{row["kept"]} 剔除{row["removed"]}帧；'
                             f'保留X/Y标准差 {row["std"][0]:.5f}/{row["std"][1]:.5f}')
            if channel in self.amplification:
                response=self.amplification[channel]
                lines.append(f'特征X增加0.001→画布ΔX/ΔY {response[0][0]:+.1f}/{response[0][1]:+.1f}px；'
                             f'特征Y增加0.001→{response[1][0]:+.1f}/{response[1][1]:+.1f}px（仿射响应，非实测精度）')
            if channel in self.signals:
                signal=self.signals[channel]
                lines.append('纵向特征中位（中/左上/右上/右下/左下）：'+
                             ', '.join(f'{v:.5f}' for v in signal['centers']))
                lines.append('各点纵向P90-P10波动：'+', '.join(f'{v:.5f}' for v in signal['spreads']))
                lines.append(f'下排减上排特征差：{signal["down_minus_up"]:+.5f}（不单独作为通过判据）')
            if channel=='average' and channel in self.training:
                lines.append(training_quality(self.training[channel])[1])
            if channel in self.errors:
                lines.append('不可评估：'+self.errors[channel])
            for stage,rows in [('训练',self.training.get(channel,[])),
                               ('验证',self.validation.get(channel,({},[]))[1])]:
                if rows:
                    lines.append(stage+'：')
                for row in rows:
                    x,y=row['axis_abs_px']
                    lines.append(f'点{row["point"]} 中位{row["median"]:.2%} 最大{row["max"]:.2%}'
                                 f'；X/Y {x:.1f}/{y:.1f}px；偏移{row["bias_px"][0]:+.1f}/{row["bias_px"][1]:+.1f}px')
                    lines.append('  头部相对基准 中位：'+', '.join(f'{v:+.3f}' for v in row['head_delta'])+
                                 '；最大绝对：'+', '.join(f'{v:.3f}' for v in row['head_max_abs']))
            if channel in self.validation:
                metrics=self.validation[channel][0]
                lines.append(f'验证总体中位{metrics["median_error"]:.2%} / 最大{metrics["max_error"]:.2%}；'
                             '仅供对照，不替换当前映射')
        if not self.training:
            lines.append('\n尚无本次五点训练诊断；加载旧档案不能恢复当时的分眼数据。')
        return '\n'.join(lines)
