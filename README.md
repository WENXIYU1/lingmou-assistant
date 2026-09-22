# 灵动视眸 / Lingmou Assistant

阶段：第一步安全核心与模拟练习原型，**不是已完成的眼控软件或参赛安装包**。

本版已实现本地暂停/跟踪状态、双轴增益与死区、时间相关平滑、持续动作一次触发、只允许练习事件的执行器、练习设置校验与备份。界面沿用CustomTkinter，输入为鼠标模拟特征和模拟眨眼，不采集摄像头，也不注入系统鼠标事件。

## 运行

建议Python 3.12，Windows环境。核心无第三方依赖，界面依赖见固定版本清单。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m apps.desktop --check
.\.venv\Scripts\python.exe -m apps.desktop
```

练习界面不代表五点校准。恢复后首个有效样本重新对齐中心；移动鼠标模拟输入，大目标命中次数仅用于调试，不是眼控准确率。离开画布只保持模拟位置，便于点击控制按钮；用“模拟丢失”按钮测试信号中断，恢复信号后仍需明确启用。Esc和暂停按钮只影响本程序。

```powershell
python -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.ui_smoke
```

练习参数保存在用户本地数据目录的`LingmouAssistant/practice-settings.json`；不代表用户校准档案。损坏配置不会自动覆盖，保存前需用户处理原文件。

## 开发入口

- [AGENTS.md](AGENTS.md)：后续修改约束。
- [实施与验收规则](docs/开发实施与验收规则.md)：需求ID、边界及验收。
- [v2方案](灵动视眸改进方案_v2_本地模型与云端协同.md)：产品路线。
- [第一步报告](docs/第一步优化报告.md)：已验证、未验证与风险。
- [来源与第三方说明](THIRD_PARTY_NOTICES.md)。

## 下一阶段

确认视觉模型来源及许可，接入真实摄像头的练习适配器；实现五点校准、独立验证和个人档案，然后再评估系统控制。PDF、语音、云端DeepSeek、本地小模型均未在本阶段实现。

远程库设为私有，研发中的代码不等于获得比赛资格。旧项目复用资格、官方材料和AI辅助开发披露仍需落实。
