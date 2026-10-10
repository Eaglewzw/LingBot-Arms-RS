# LingBot-Arms-RS

面向 **Seeed Studio reBot B601-RS** 的 LingBot-VLA 2.0 推理与控制工程，支持单臂抓取的部署流程，并为双臂协作任务预留接口。

项目基于 [reBot VR 遥操作工程](https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-vr)，复用其 RS 驱动与 MIT 控制链路。计划通过 VR 采集真机示范、完成 RS 后训练，再部署到机械臂。

> 当前已通过软件测试，真实模型推理和机械臂联调仍在推进。抓取与双臂叠块能力需要后训练及实测验证。

## 功能

- **独立推理服务**：通过 HTTP 连接 LingBot-VLA，本机或远端运行模型。
- **单臂控制**：右臂 RS、外部相机和右腕相机，支持动作块执行。
- **执行约束**：关节限位、速度与加速度限制、观测超时及过期动作检查。
- **双臂接口**：共享观测、联合动作输出，提供叠三个方块的配置入口。
- **运行记录**：保存状态、动作、推理延迟和故障日志。

| 模式 | 用途 | 电机动作 |
|---|---|---|
| `mock` | 无硬件的软件链路测试 | 无 |
| `shadow` | 真实相机与关节反馈，只读检查模型输出 | 无 |
| `live` | RS 后训练模型的单臂实机执行 | 需显式 `--execute` |

`mock` 不包含物理仿真；双臂目前仅支持 `mock` 和 `shadow`。

## 快速开始

软件测试使用 Python 3.10+；真实 RS 控制需要 Python 3.12+ 及 VR 工程依赖。

```bash
git clone https://github.com/Eaglewzw/LingBot-Arms-RS.git
cd LingBot-Arms-RS
python -m pip install -e '.[test]'

# 检查配置
rs-lingbot --config configs/single_right.yaml --check

# 运行单臂软件模拟，不需要模型、GPU 或机械臂
rs-lingbot --config configs/single_right.yaml --mode mock --duration 10

# 运行双臂软件模拟
rs-lingbot --config configs/dual_stack_three.yaml --mode mock --duration 10
```

## 模型部署

模型端使用 LingBot-VLA 的推理环境；控制端安装相机和 RS 硬件依赖。准备匹配的 checkpoint、训练配置、归一化统计和 Qwen3-VL 资源后，启动服务：

```bash
rs-lingbot-server --bundle /path/to/rs_bundle --lingbot-repo /path/to/lingbot-vla-v2 --qwen-path /path/to/Qwen3-VL-4B-Instruct --precision bf16
```

配置相机和机械臂后，以只读模式连接：

```bash
rs-lingbot --config configs/single_right.yaml --mode shadow --policy-url http://127.0.0.1:9331 --duration 30
```

完整的依赖安装、模型打包、硬件配置及实机步骤见 **[部署指南](docs/deployment.md)**。RoboTwin 权重可用于诊断，直接控制 RS 需要对应的真机后训练模型。

## 项目结构

```text
configs/          单臂、双臂与数据映射配置
src/rs_lingbot/   推理服务、观测采集、动作执行和硬件适配
tests/            软件与接口测试
docs/             部署和使用说明
```

## 测试

```bash
python -m pytest -q
```

测试使用模拟硬件，覆盖动作单位、轨迹约束、超时、HTTP 通信和检查点校验，不代表任务成功率。

## 实机使用说明

运行前需完成零位校准、夹爪行程测量和低速验证。当前不具备碰撞检测；退出时请求失能，机械臂可能下落，须准备机械支撑及实体急停。详细行为见[部署指南](docs/deployment.md#单臂真机执行)。
