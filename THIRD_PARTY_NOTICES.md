# 来源、依赖与开发披露

参考来源：https://github.com/lyk05212007/eyemouse 。本地保留下载的原始源码供静态审阅，不作为本次上传内容。原代码许可尚未确认，本仓库未附加开源许可证或宣称全部算法原创。

本阶段依据参考代码中观察到的交互问题实现独立模块；未复制原始摄像头循环。共同使用的Python和CustomTkinter技术栈不等于原项目整体已迁移。后续复制或改造参考代码前必须确认授权并记录来源版本。

界面依赖：CustomTkinter 5.2.2、darkdetect 0.8.0、packaging 24.2。发行时保留相应发行包许可证并再次核验。第二阶段已在本地下载视觉模型，未将权重上传、打包或再分发；尚未下载语言模型。

## 第二阶段视觉来源与隐私记录

- [官方Face Landmarker指南](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)：本项目使用VIDEO模式与递增时间戳，在独立进程内推理，不在UI线程运行摄像头循环。
- [官方关键点连接定义](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py)：虹膜环469–472和474–477；用环均值而非单一474点估计虹膜位置。
- [官方固定v1模型示例来源](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/face-landmarker.ts)：模型地址、SHA256及本地路径保存在`models/manifest/face_landmarker.json`。
- MediaPipe框架为Apache-2.0；[FaceMesh V2官方模型卡](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)声明Apache-2.0。不可把文档/示例代码许可自动当成全部资源许可；完整bundle其他组件的随包声明、NOTICE及发行义务在正式打包前继续审核。当前只作本地开发使用，不再分发权重。
- [官方遥测讨论#6291](https://github.com/google-ai-edge/mediapipe/issues/6291)：0.10.35空白输入测试出现clearcut连接失败日志，未使用人脸或摄像头数据。官方维护者表示新版本含遥测且无关闭API，并表示不上传输入数据。基于项目离线原则已切换到0.10.21，加载器拒绝其他版本。旧版空白推理未出现同类日志，但不将此视为完整网络审计；不自动升级。发布前仍需离线网络审计与依赖安全复核。
- 音频库是MediaPipe的依赖，本项目没有调用麦克风或音频采集。所有实时摄像头画面仅在进程内处理，不进入日志、档案或网络请求。

开发使用Codex进行代码编写与测试辅助。具体模型标识、版本与AI编写代码比例待按真实会话记录和统一统计口径整理；不得用猜测值或本声明代替正式参赛披露。团队仍需审阅、验证并说明自身设计贡献。
