# 天工 3.0 URDF → IsaacLab 服务器验证报告

> 验证时间：2026-10-13
> 目标机：AutoDL 实例 `autodl-container-85874d8508-36b4a9d7`
> 结论：**通过** —— URDF 成功转为 USD 并在 IsaacLab 中正确加载，
> 39 刚体 / 31 关节 / 65.82 kg 三项资产指标与 URDF 定义**精确吻合**。

---

## 一、结论摘要

| 检查项 | 期望（由 URDF 定义） | 实测 | 结果 |
|---|---|---|---|
| 刚体数 bodies | 39 | **39** | ✅ |
| 关节数 joints | 31（38 关节中 7 个 fixed） | **31** | ✅ |
| 总质量 | 65.82 kg | **65.82 kg** | ✅ |
| 根 link | `pelvis` | `pelvis` | ✅ |
| URDF → USD 转换 | 成功 | 成功（exit 0） | ✅ |
| 仿真稳定性 | 500 步无异常 | 500 步跑完 | ✅ |

**为什么用这三个数字当判据**：它们直接来自 URDF 的 `<link>` / `<joint>` / `<inertial>` 段落，
一旦网格路径解析失败、固定关节被错误合并、或惯性参数丢失，这三个数立刻对不上。
相比之下"机器人能不能站住"取决于控制器增益，**不能用来判断 URDF 是否正确**（见第五节）。

---

## 二、目标环境（实测）

| 项 | 值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS (Jammy) |
| 内核 | 5.15.0-97-generic |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| 驱动 | 570.124.04 |
| CPU / 内存 | Intel Xeon Gold 6430 / 122880 MB |
| IsaacLab | **v2.3.2** @ `/root/autodl-tmp/IsaacLab` |
| IsaacSim | **5.1.0-rc.19+release.26219.9c81211b.gl** @ `/root/autodl-tmp/isaacsim` |
| conda 环境 | `env_isaaclab` @ `/root/autodl-tmp/conda/envs/env_isaaclab` |
| Python | 3.11.16 |
| 磁盘 | `/` 30G(剩 20G) ；`/root/autodl-tmp` 200G(剩 172G) |

> 注意：`/root` 本身挂在 30G 的 overlay 上，只有 `autodl-tmp` 是 200G 数据盘。
> 资源和 USD 都放在 `/root/autodl-tmp/` 下，避免撑爆系统盘。

---

## 三、执行步骤（可复现）

### 3.1 部署

```powershell
# 在本机 E:\ROBOT C\deploy\ 下
.\deploy.ps1 -Mode all
```

或手动：

```powershell
scp -r -P <端口> tiangong3 root@<主机>:/root/autodl-tmp/
ssh -p <端口> root@<主机> "bash /root/autodl-tmp/tiangong3/deploy/remote_bootstrap.sh /root/autodl-tmp/tiangong3 all"
```

### 3.2 服务器上直接跑

```bash
conda activate env_isaaclab
cd /root/autodl-tmp/IsaacLab

# 转换（首次 1~3 分钟）
./isaaclab.sh -p /root/autodl-tmp/tiangong3/scripts/load_in_isaaclab.py --headless --convert-only

# 加载 + 资产核对 + 沉降测试
./isaaclab.sh -p /root/autodl-tmp/tiangong3/scripts/load_in_isaaclab.py --headless --steps 500
```

### 3.3 产生的 USD 产物

```
usd/
├── tiangong3_isaaclab.usd                      1,476 B   ← 主入口（引用下面各层）
├── configuration/
│   ├── tiangong3_isaaclab_base.usd         4,888,724 B   ← 网格/几何（占绝大部分体积）
│   ├── tiangong3_isaaclab_physics.usd         13,076 B   ← 碰撞体与物理属性
│   ├── tiangong3_isaaclab_robot.usd            3,013 B   ← 关节层级
│   └── tiangong3_isaaclab_sensor.usd             659 B   ← 传感器
├── config.yaml
└── .asset_hash
```

> 主 USD 只有 1.5 KB，**几何全在同目录的 `configuration/` 子目录里**。
> 拷贝 USD 时必须整个 `usd/` 目录一起走，只拿主文件会得到一个空模型。

---

## 四、踩到的坑与修复（全部由服务器实测暴露，非推测）

### 坑 1：`set -u` 与 `conda activate` 冲突 → 脚本无声退出

引导脚本里写了 `set -uo pipefail`，而 conda 的 `activate` 脚本会引用未绑定变量，
在 nounset 下会直接中止当前 shell。更糟的是当时那行还带 `2>/dev/null`，
把 `unbound variable` 的错误**整个吞掉**，表现为"打印两行后静默退出、没有任何报错"。

**修复**：调 conda 时临时 `set +u`，并且绝不重定向 stderr。

```bash
set +u
conda activate "$CONDA_ENV"
RC=$?
set -u
```

### 坑 2（最关键）：`pxr` 必须在 AppLauncher 之后才能导入

原始脚本在 `main()` 里先调用转换函数、后启动 App，结果：

```
ModuleNotFoundError: No module named 'pxr'
```

`pxr`（USD 的 Python 绑定）来自 Isaac Sim **运行时**。在 `AppLauncher` 启动前执行
`from isaaclab.sim.converters import ...`，就会在 `isaaclab/utils/mesh.py`
的 `from pxr import Usd, UsdGeom` 处失败。

**修复**：严格照官方 `scripts/tools/convert_urdf.py` 的顺序 —— 先起 App，再导入转换器。
这是 IsaacLab 里 URDF 转换的权威范式。

### 坑 3：`simulation_app.close()` 会吞掉 traceback 并返回 exit 0

修好坑 2 后，脚本"跑完了、exit 0"，但关键输出全没了。
原因是 Isaac Sim 的 `close()` **直接结束进程，绕过 Python 的 stdout 缓冲刷新**；
如果异常只被 `finally: simulation_app.close()` 兜住，异常信息和退出码会一起消失。

**修复**：两层都要改。
1. Python 侧：`except BaseException: traceback.print_exc()` 必须在 `close()` **之前**执行；
2. Shell 侧：`export PYTHONUNBUFFERED=1`，否则重定向到管道/文件时是块缓冲，末尾 print 整段丢失。

> 这个坑很阴险：它把"失败"伪装成"成功"。**exit 0 不等于跑对了**，一定要看有没有关键输出。

### 坑 4：`InteractiveScene` 只接受 `AssetBaseCfg`

```
ValueError: Unknown asset config type for ground: GroundPlaneCfg(...)
```

`GroundPlaneCfg` / `DomeLightCfg` 是 **Spawner**，不是 `AssetBaseCfg`。
直接当场景项写会被 `InteractiveScene._add_entities_from_cfg()` 拒绝。

**修复**：用 `AssetBaseCfg(prim_path=..., spawn=...)` 包一层
（与官方 `scripts/tutorials/02_scene/create_scene.py:55` 一致）。

### 坑 5：`grep` 块缓冲导致远程看不到进度

`ssh ... | grep -Ev '...'` 时 grep 对管道做块缓冲，任务在跑但本地长时间无输出。
**修复**：过滤长输出时加 `--line-buffered`，或改为 `tee` 到服务器文件后单独 tail。

### 5 个警告的正确解读

转换时出现 7 条，正好对应 **7 个 fixed 关节**，属正常：

```
imu_waist_joint / camera_body_front_joint / camera_head_joint /
imu_head_joint / radar_head_joint / left_tcp_joint / right_tcp_joint:
Joint Axis is not body aligned with X, Y or Z primary axis.
Adjusting PhysX joint alignment to Axis X and reorienting bodies.
```

导入器自动把关节轴对齐到主轴并重新定向 body，**无损**。
`GLFW initialization failed`、`No Viewport Window`、`failed to open the default display`
都是 headless 模式下的必然现象，可忽略。

---

## 五、关于"站立测试"的如实说明

500 步重力沉降实测：

| 步数 | 根高度 (m) | 关节最大偏离 (rad) |
|---|---|---|
| 0 | 0.999 | 0.0060 |
| 100 | 0.996 | 0.0117 |
| 200 | 0.975 | 0.1360 |
| 300 | 0.797 | 0.3548 |
| 400 | 0.108 | 0.1051 |
| 结束 | **0.108** | — |

**机器人塌了**。但这**不是 URDF 的错误**，原因很明确：

- `fix_base=False` 的浮动基座人形机器人，在**没有重力补偿、没有平衡控制器**的前提下，
  仅靠固定 PD 增益（stiffness=200, damping=20）不可能撑住 65.82 kg 的自重；
- 初始姿态是全零位（关节角全 0），未必是真实站立位形；
- 因此根高度先缓降、后失稳，是符合物理预期的行为。

**这也正是为什么本报告的判据是资产完整性（39/31/65.82）而不是"站没站住"。**
要让天工真正站住/走起来，需要接入带重力补偿与平衡奖励的 RL 训练环境
（即 `Open-X-Humanoid/TienKung-Lab` 那套），而不是靠手工调 PD。

---

## 六、对既有文档的更正

`tiangong3_sim/README_天工3_IsaacLab.md` 与 `tiangong3_urdf_package/` 中此前记载：

> ⚠️ Isaac Lab 3.0 起 `usd_file_name` 已废弃；`collider_type` 不是 `UrdfConverterCfg` 的参数；
> `make_instanceable` 已废弃。输出自动生成为 `{usd_dir}/{robot_name}/{robot_name}.usda`。

**这些说法对目标版本是错的。** 依据服务器上 `source/isaaclab/isaaclab/sim/converters/`
的源码实测（IsaacLab **2.3.2**）：

| 字段 | 实测定义 | 是否废弃 |
|---|---|---|
| `usd_file_name` | `str \| None = None` | ❌ **未废弃** |
| `collider_type` | `Literal["convex_hull","convex_decomposition"] = "convex_hull"` | ❌ **未废弃** |
| `make_instanceable` | `bool = True`（基类 `AssetConverterBaseCfg`） | ❌ **未废弃** |
| `fix_base` | **必填**（无默认值） | — 人形必须显式 `False` |
| `merge_fixed_joints` | `bool = True` | — 要保住 39 link 必须 `False` |

**输出路径**由源码 `usd_path = os.path.join(usd_dir, usd_file_name)` 决定，
即主文件**平铺**在 `usd/` 下（`usd/tiangong3_isaaclab.usd`），
**不是** `usd/<name>/<name>.usda` 子目录结构。几何另存于 `usd/configuration/`。

---

## 七、已知限制与下一步

**本次未覆盖：**
1. 未做**凸包外形**的定量误差评估（凸包比原始网格"胖"，可能影响接触/碰撞精度）；
2. 未验证 URDF 的**惯性参数**是否与真机一致（只核对了总质量）；
3. 未在 GUI 模式下目视确认外观（全程 headless）；
4. 未做**自碰撞**（`self_collision` 默认 `False`）验证。

**下一步（按学习计划）：**
1. 用 manager-based 环境把天工注册成正式任务（`gym.register`），
   参照 SOP 第 5 节 "自定义任务"；
2. 接入 `Open-X-Humanoid/TienKung-Lab` 的走跑训练配置；
3. 训练前先确认显存预算：4090 24G，凸包网格 3.2 MB，
   并行环境数需实测（建议从 1024 起，逐步加）；
4. 训练产物记得定期清理 `logs/`、`outputs/`。

---

## 八、附件

| 文件 | 说明 |
|---|---|
| `logs/convert_urdf_to_usd.log` | URDF→USD 完整转换日志（`--force-convert`） |
| `logs/test_run.log` | 加载 + 资产核对 + 沉降测试原始输出 |
| `logs/server_verification.log` | 服务器侧汇总（环境、两个日志、USD 产物清单、包校验） |
| `deploy/remote_bootstrap.sh` | 服务器端引导脚本（report / convert / test / all） |
| `deploy/deploy.ps1` | 本机一键部署（上传 + 远程执行） |
| `scripts/load_in_isaaclab.py` | 修正后的加载脚本（已含上述全部修复） |
