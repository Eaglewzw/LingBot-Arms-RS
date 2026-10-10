# LingBot-Arms-RS

面向 Seeed Studio reBot B601-RS 的 LingBot-VLA 推理与控制工程。

本工程将 LingBot-VLA 2.0 的动作块接入 Seeed reBot B601-RS：右臂 `can0`、一台外部相机、一台右腕相机。已有单臂模型服务、相机观测、异步推理、动作执行约束、RS MIT 驱动适配和模拟测试。**这是一版未经真机验证的控制软件，不是已经学会抓取或叠块的策略。**

代码复用 VR 项目的 `RSFollower`、`MITCommandDispatcher` 和 `ManagedFollower`，不重写 CAN 协议，不调用被禁止的 `RSFollower.send_action()`，不修改原 LingBot 工程。

参考版本：[Seeed-Projects/lerobot-teleoperator-rebot-vr，79d1d3a](https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-vr/tree/79d1d3aec2283c625a72128be0a5fcf07d8a8348)。若更新该依赖，应重新核验硬件适配器接口、零位及关闭行为。

## 当前可以做什么

| 模式 | 相机/反馈 | 模型 | 电机行为 |
|---|---|---|---|
| `mock`，默认 | 合成图像、模拟关节 | 小幅测试动作，或指定 HTTP 模型 | 不导入硬件依赖，不连接 CAN |
| `shadow` | 真实相机、只读 RS 反馈 | HTTP LingBot 服务 | 不切换模式、不使能、不写零位、不发动作、不发失能 |
| `live --execute` | 真实相机、RS 反馈 | 与配置匹配的 RS 后训练模型 | 单臂受限 MIT 执行；退出请求失能 |

`mock` 只验证软件链路，合成图像和小幅动作不能说明抓取成功率。`shadow` 评估当前反馈位置上下一步受限命令，不模拟真实接触、碰撞或完整轨迹。

现有 **18K 是 RoboTwin 权重**。可构建只读诊断包检查输入、输出及推理延迟，但不能将改名、换归一化或补零当作 RS 后训练。`live` 拒绝未声明 RS 后训练的模型。`--rs-finetuned` 是部署者对权重来源的声明，不是软件验证了模型能力。

## 快速运行：不需要机械臂、GPU 或 checkpoint

以下命令均在 `LingBot-Arms-RS/` 目录执行。软件模拟支持 Python 3.10+；真实 RS 控制使用 VR 项目要求的 **Python 3.12+** 环境。

```bash
python -m pip install -e '.[test]'
rs-lingbot --config configs/single_right.yaml --check
rs-lingbot --config configs/single_right.yaml --mode mock --duration 10
rs-lingbot --config configs/dual_stack_three.yaml --mode mock --duration 10
python -m pytest -q
```

不安装本项目也可运行：

```bash
PYTHONPATH=src python3 -m rs_lingbot.cli --config configs/single_right.yaml --mode mock --duration 10
```

模拟的双臂只验证 `[左臂7维, 右臂7维]` 的共同观测和动作顺序，**不执行叠块任务、不证明双臂协调成功**。

## 目录与执行流程

```text
LingBot-Arms-RS/
├── configs/single_right.yaml       # 右臂＋外部相机＋右腕相机
├── configs/dual_stack_three.yaml   # 双臂联合模型的 mock/shadow 配置
├── configs/robot_rs_single.yaml    # LingBot 数据特征映射
├── configs/robot_rs_dual.yaml      # 左右联合特征映射
├── src/rs_lingbot/
│   ├── contract.py                 # 单位、维度、关节顺序、夹爪转换
│   ├── cameras.py                  # 后台 RGB 采集、时间和新鲜度检查
│   ├── policy.py                   # HTTP 客户端、单个在途推理请求
│   ├── server.py                   # 独立 LingBot GPU 服务
│   ├── bundle.py                   # 检查权重完整性、准备配置与统计
│   ├── safety.py                   # 关节位置、速度、加速度与跟踪约束
│   ├── hardware.py                 # 复用 VR 项目的 RS/MIT/关闭流程
│   └── runtime.py                  # 50 Hz 控制循环、过期丢弃、日志
└── tests/                         # 模拟驱动、故障、网络、数据契约测试
```

```text
外部/腕部相机 ─┐
RS 实际反馈 ───┼─> 时间检查与观测打包 ─> 异步 HTTP 推理 ─> 动作块
文字任务 ──────┘                                         │
                                                       ↓
RS 电机 <─ MITCommandDispatcher <─ 轨迹约束 <─ 按观测时间选取有效动作
```

模型不运行在电机循环里。动作索引相对本机采集时间计算：网络/推理期间已经过去的动作会被跳过，整块过期则拒绝。无可用动作时先减速；达到等待超时则结束会话。默认参数适合起步验证，需要根据实测推理延迟调整，不能把网络超时任意放大来掩盖旧观测。

## 准备模型部署包

推理与机械臂最好使用两个独立 Python 环境。控制端安装本项目、OpenCV、已验证版本的 VR 插件；模型端使用当前 LingBot 工程的推理依赖，并安装本项目。HTTP 协议隔离了 PyTorch 与硬件 SDK 的版本要求。

控制环境示例：

```bash
python -m pip install -e '.[camera]'
python -m pip install 'lerobot_teleoperator_rebot_vr @ git+https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-vr.git@79d1d3aec2283c625a72128be0a5fcf07d8a8348'
```

每个部署包需要：

1. 完整 `hf_ckpt`，含所有 safetensors 分片和索引。
2. 该次训练的真实 `lingbotvla_cli.yaml` 或完整 `model/data/train` 配置，不能随便使用公开示例配置代替。
3. 该机器人数据实际计算的 `norm_stats.json`。单臂 arm/effector 统计维度为 6/1；双臂为 12/2。
4. 与模型匹配的本地 Qwen3-VL 配置、tokenizer 和 processor 资源。

准备 **RS 单臂后训练** 权重：

```bash
python -m rs_lingbot.bundle \
  --checkpoint /path/to/rs_single_checkpoint/hf_ckpt \
  --training-config /path/to/rs_single_training/lingbotvla_cli.yaml \
  --norm-stats /path/to/rs_single_training/norm_stats.json \
  --config configs/single_right.yaml \
  --output bundles/rs_single \
  --rs-finetuned
```

准备 **18K 的只读诊断包**：

```bash
python -m rs_lingbot.bundle \
  --checkpoint /path/to/global_step_18000/hf_ckpt \
  --training-config /path/to/actual_18k_training.yaml \
  --norm-stats /path/to/actual_18k_norm_stats.json \
  --config configs/single_right.yaml \
  --output bundles/18k_diagnostic \
  --robotwin-diagnostic
```

请从生成该 checkpoint 的训练任务中取得对应配置与统计文件。仓库不包含模型权重、个人训练配置或数据集；示例中的路径需要替换为自己的路径。

诊断模式在 RoboTwin 双臂输入中补零填入缺失机械臂，输出提取所选一侧；缺失腕部图像通过 LingBot 原有图像掩码处理。**这不是可靠的跨本体动作转换，只用于检查管线，始终禁止真机动作。** 超出 RS 限位的预测仍会被拒绝。

打包会检查分片是否齐全、索引是否匹配、文件长度是否符合 safetensors 头部；这不替代整个下载任务的 SHA-256 校验。权重使用符号链接以避免重复占用磁盘，所以原权重目录必须保留；迁移到另一台主机时应重新打包。输出目录必须不存在。

启动 GPU 模型服务（在模型环境执行）：

```bash
python -m rs_lingbot.server \
  --bundle bundles/rs_single \
  --lingbot-repo /path/to/lingbot-vla-v2 \
  --qwen-path /path/to/Qwen3-VL-4B-Instruct \
  --precision bf16
```

本机 5070 Ti 16 GB 能否承载 BF16 及其推理延迟尚未实测；FP32 权重无法按当前整模型加载方式装入该显存。服务默认关闭 `torch.compile` 并绑定 `127.0.0.1:9331`。如使用服务器推理，应通过 SSH 转发访问，HTTP 服务不带身份认证，不能暴露到公网。

## 真实相机与只读 RS 检查

编辑 `configs/single_right.yaml`：

- 把两个 `device` 改成真实相机设备，优先使用稳定的 `/dev/v4l/by-id/...` 路径。默认采集 V4L2 RGB 图像。
- 保持与采集/后训练相同的视角、裁剪、颜色、分辨率和相机顺序。
- `robot_id`、`calibration_dir` 与 VR 校准使用的值一致；代码不会写机械零位或跳过校准。
- 用实机测量的夹爪端点替换示例值；确认后才设置 `gripper_endpoints_verified: true`。

```bash
rs-lingbot --config configs/single_right.yaml \
  --mode shadow --policy-url http://127.0.0.1:9331 --duration 30
```

进入实时循环前，会先只读关节反馈并做一次模型预热；这一步不使能电机。相机时间戳是本机读帧完成时间，并非传感器曝光时间；异步 USB 摄像头不构成硬件同步，部署前需要检查缓冲延迟和视觉/动作对齐。

## 单臂真机执行

必须先完成 RS 数据后训练、确认动作方向/单位/夹爪映射、验证相机与推理延迟，并在明确的工作空间内做低速联调。配置的关节范围比 URDF 窄，启动前需由操作者把机械臂放到范围内的稳定准备姿态；本工程不会自动从零位运动到准备位。

```bash
rs-lingbot --config configs/single_right.yaml \
  --mode live --execute \
  --policy-url http://127.0.0.1:9331 --duration 10
```

停止方式为 Ctrl+C、SIGTERM，或在工程下创建 `STOP` 文件：

```bash
touch STOP
```

已存在的 STOP 文件会阻止下一次启动；排查后由操作者移走它。本版故障/退出会通过 VR 的 `ManagedFollower` 请求失能，**不会自动回零，失能后机械臂可能下落，应预先确保机械支撑和实体急停可用**。关节限制不等于笛卡尔碰撞检测；当前无桌面/物体/人体碰撞检测，也不能保证进程被强杀、掉电或 CAN 故障时的硬件状态。失能报告保留“发出请求”和“确认失能”的区别，不把缓存状态当作失能证明。

## 日志和演示边界

每次运行在 `runs/时间戳/` 保存：模型身份和契约、完整运行配置、观测状态、受限命令、模型动作块、推理延迟、故障与退出结果。日志同步写磁盘，慢盘可能触发控制超时；当前工程不是硬实时系统。

这些日志不包含逐帧图像，不是可直接训练的完整 LeRobot 数据集。VR 示范采集应另行同步保存图像、实际状态、最终下发目标、任务描述及 episode 边界。单臂数据每帧 7 维，双臂每帧 14 维；模型内部 55 维由原工程 `FeatureTransform` 填充和掩码，不把缺失臂的零值伪装成真实 RS 示范。

**双臂叠三个方块的后续工作：** 用同步双臂示范训练一个联合策略，输入共享外部相机与左右腕相机，一次输出左右动作；补充基座外参、双臂/桌面碰撞检查、共同故障停止及实机验证后才开放双臂执行。`dual_stack_three.yaml` 已提供相同协议的 mock/shadow 入口，`live` 当前明确拒绝双臂。工程没有硬编码“看见方块就抓取”的规则，也没有自动判定三块叠放成功；任务能力来自后训练权重，演示需独立评估。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q
```

测试覆盖：弧度/角度与夹爪换算、接口不匹配、速度/加速度约束和反向制动、非法动作与反馈、动作过期、默认不执行、模拟单/双臂循环、HTTP 图像及动作往返、模拟 RS 使能/关闭路径、检查点截断与索引不匹配、RS 与 RoboTwin 统计维度隔离。

测试不连接硬件、不测量模型成功率。第一次提交的真实 GPU 推理、相机和机械臂联调仍待完成。
