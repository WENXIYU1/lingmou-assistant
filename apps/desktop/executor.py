"""Only a practice sink exists. There is deliberately no OS control adapter."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PracticeAction:
    action_id: str
    epoch: int
    kind: str


class PracticeExecutor:
    def __init__(self, controller):
        self.controller = controller
        self.epoch = controller.epoch
        self.seen = set()
        self.events = deque(maxlen=100)

    def execute(self, action):
        if self.epoch != self.controller.epoch:
            self.seen.clear()
            self.epoch = self.controller.epoch
        if (not self.controller.can_practice or action.epoch != self.epoch
                or action.kind not in ('left', 'right')
                or not isinstance(action.action_id, str) or not 0 < len(action.action_id) <= 128
                or action.action_id in self.seen):
            return False
        if len(self.seen) >= 10000:
            self.controller.fail('本次练习动作数量超限，请重新启动')
            return False
        self.seen.add(action.action_id)
        self.events.append(action.kind)
        return True

    def execute_system(self, *args, **kwargs):
        raise PermissionError('本阶段未开放系统级控制：需要完整校准和独立验收')
