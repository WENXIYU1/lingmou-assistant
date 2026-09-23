"""Bounded UI-received observation audit, not a sensor FPS measurement."""
from collections import deque, Counter
from statistics import median
import math


class CaptureAudit:
    def __init__(self):
        self.samples=deque(maxlen=300)

    def clear(self):
        self.samples.clear()

    def feed(self, frame, received_at):
        if not math.isfinite(received_at) or not math.isfinite(frame.timestamp):
            return
        if self.samples and received_at<=self.samples[-1][0]:
            return
        age=received_at-frame.timestamp
        valid=frame.usable() and 0<=age<=.25
        reason='有效' if valid else ('过期/时间异常' if frame.usable() else frame.reason)
        widths=frame.eye_widths_px if valid else ()
        self.samples.append((received_at,valid,reason,frame.capture_size,widths,age))

    def report(self, now):
        rows=[r for r in self.samples if 0<=now-r[0]<=10]
        lines=['采集质量审计（最近10秒，最多300个界面接收结果）',
               '无需五点校准；不保存、不上传，不显示原始画面。',
               '以下是界面接收结果统计；包含推理/队列影响，不是硬件帧率。']
        if not rows:
            return '\n'.join(lines+['尚无最近数据：请手动开启摄像头。'])
        duration=now-rows[0][0]
        valid=sum(r[1] for r in rows)
        lines.append(f'实际最近图像尺寸：{rows[-1][3][0]}×{rows[-1][3][1]}；观察时长{duration:.1f}秒')
        if duration>=1:
            # Exclude first sample at the interval boundary from rate numerators.
            lines.append(f'界面结果到达率：{(len(rows)-1)/duration:.1f}/秒；'
                         f'有效结果率：{sum(r[1] for r in rows[1:])/duration:.1f}/秒')
        else:
            lines.append('时长不足1秒，暂不估计速率')
        lines.append(f'收到{len(rows)}帧；有效{valid}；有效占比{valid/len(rows):.1%}（仅接收样本）')
        lines.append(f'最近结果距今{now-rows[-1][0]:.2f}秒；超过0.25秒不代表当前仍可跟踪')
        widths=[r[4] for r in rows if len(r[4])==2 and all(math.isfinite(v) and v>0 for v in r[4])]
        if widths:
            lines.append('有效帧眼角间距中位：左'+f'{median(w[0] for w in widths):.1f}px / 右'+
                         f'{median(w[1] for w in widths):.1f}px（不是虹膜直径）')
        else:
            lines.append('眼部像素宽度不可用：没有带测量值的有效帧')
        lines.append('接收时帧龄中位：'+f'{median(r[5] for r in rows)*1000:.1f}ms（含读取/推理/队列时间）')
        reasons=Counter(r[2] for r in rows if not r[1])
        lines.extend(f'无效原因：{reason or "未说明"} ×{count}' for reason,count in reasons.items())
        lines.append('队列中被覆盖的结果未计入；不能据此推断所有采集帧的有效率。')
        return '\n'.join(lines)
