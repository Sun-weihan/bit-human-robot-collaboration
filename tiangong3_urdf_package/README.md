# 天工 3.0 URDF 资源包

> 解压即用。**动手前请先读完「二、三个坑」** —— 跳过那部分大概率会踩雷。

- 机器人：天工 3.0（TienKung 3.0）人形机器人
- 来源：https://github.com/Open-X-Humanoid/TienKung_URDF （目录 `tiangong3_urdf/`）
- 打包日期：2026-10-13
- 用途：Isaac Lab / MuJoCo / ROS 2 仿真

---

## 一、包内有什么

```
tiangong3_urdf_package/
├── README.md                      <- 本文件
├── urdf/
│   ├── tiangong3_isaaclab.urdf    ★ 用这个（路径已修正）
│   └── tiangong3.urdf               官方原始版（路径失效，仅作对照）
├── meshes_convex/                 39 个 STL，凸包版，3.23 MB
├── scripts/
│   ├── verify_urdf.py             自检（不装 Isaac Lab 也能跑）
│   ├── load_in_isaaclab.py        URDF→USD 转换 + 场景 + 站立测试
│   └── fetch_assets.py            重新下载（一般用不到）
└── docs/
    └── README_detailed_isaaclab.md  详细版说明（含 URDF 原理、故障排查）
```

**实测参数**（可直接用于交叉核对）：

| 项 | 值 |
|---|---|
| link 数 | 39 |
| joint 数 | 38（revolute 31 + fixed 7） |
| 可动自由度 | 31 |
| `<inertial>` | 39 个，全齐 |
| 总质量 | **65.82 kg** |
| base link | `pelvis` |
| 凸包三角面 | 67,574 |
| 关节限位 | 齐全（lower/upper/effort 都有） |

关节命名规范，左右用 `_l` / `_r` 后缀：

```
腿：hip_pitch / hip_roll / hip_yaw / knee_pitch / ankle_pitch / ankle_roll
腰：waist_yaw / waist_roll / waist_pitch
头：head_yaw / head_pitch
臂：shoulder_pitch / shoulder_roll / shoulder_yaw / elbow_pitch / elbow_yaw
    / wrist_pitch / wrist_roll
末端：left_tcp_link / right_tcp_link
```

---

## 二、三个坑（必读）

### 坑 1：不要用 `tiangong3.urdf`，用 `tiangong3_isaaclab.urdf`

**问题**：官方原始 URDF 里的网格路径是

```xml
<mesh filename="package://tiangong3_0_V2_urdf/meshes/pelvis.STL"/>
```

`package://` 是 ROS 的包解析协议，**在非 ROS 环境下无法解析**。
而且这个包名（`tiangong3_0_V2_urdf`）与实际仓库目录名（`tiangong3_urdf`）**不一致**，
就算装了 ROS 也找不到。

**后果**：转换时报 `Cannot find mesh file`；或者更糟 ——
**转换"成功"但模型是空的**，视口里什么都没有。这种"静默失败"最费时间。

**修复**：`tiangong3_isaaclab.urdf` 里 78 处路径已全部改为相对路径：

```xml
<mesh filename="../meshes_convex/pelvis.STL"/>
```

> 注意：改完路径后，URDF 与 `meshes_convex/` 的**相对位置不能变**。
> 如果要把 URDF 挪到别处，记得同步搬网格目录。

---

### 坑 2：这套网格是"凸包"，外形偏胖

**这点容易被忽略，但会直接影响视觉相关的工作。**

官方仓库提供两套网格：

| 目录 | 大小 | 用途 |
|---|---|---|
| `meshes/` | **136.0 MB** | 视觉渲染，外形准确 |
| `meshes_convex/` | **3.23 MB** | 碰撞计算，**外形偏胖** |

差 42 倍。本包给的是**凸包版**。

**凸包是什么**：包裹原始网格的"最小凸多面体"。它把凹进去的地方都填平了，
所以：

- ✅ 碰撞检测又快又准（只判断有没有相交，不需要精度）
- ❌ **外形不准** —— 机器人看起来会比真实零件胖一圈

**什么时候你需要换成精细版**：

- 做**视觉策略**（Diffusion Policy，第 10-12 周）→ 必须用 `meshes/`
- 要在 Isaac Lab 里看**准确外观**截图/录视频 → 用 `meshes/`
- 只要你做 RL 训练、运动学/动力学验证 → 凸包版完全够用

**换法**：改 `scripts/fetch_assets.py`：

```python
# 把
MESH_DIR = PROJECT_ROOT / "meshes_convex"
# 改成
MESH_DIR = PROJECT_ROOT / "meshes"
# 同时把 step_fetch_meshes() 里的 "meshes_convex" 改成 "meshes"
```

然后重跑（约 136 MB，视网速需几分钟）。

> 进阶：Isaac Lab 支持「视觉用精细网格 + 碰撞用凸包」的双层方案，
> 兼顾性能和外貌。做视觉任务时再考虑。

---

### 坑 3：`fix_base` 必须为 False

天工 3.0 的 URDF **没有 floating 关节**，root link 是 `pelvis`。
Isaac Lab 转换时必须显式指定：

```python
fix_base=False      # 人形机器人必须浮动基座
```

**写错的后果很迷惑人**：模型显示完全正常，但**机器人被钉死在半空中**，
一动不动。你会以为是控制参数问题，排查半天才发现是这一项。

顺带一个相关建议：`merge_fixed_joints` 建议设 `False`。
URDF 里有 7 个 fixed 关节，合并会改变 link 结构，导致你按名字找 link 时找不到。

---

## 三、怎么用

### 第 1 步：先自检（不需要 Isaac Lab）

```bash
pip install yourdfpy numpy
python scripts/verify_urdf.py
```

会检查解析、mesh 加载、质量、正运动学。**全绿再往下走**，
这样可以先把"URDF 本身有问题"这个可能性排除掉。

预期结尾：

```
✓ URDF 解析正常
✓ 全部 mesh 加载成功
✓ 质量/惯量完整
✓ 正运动学可计算
```

### 第 2 步：转 USD（需要 Isaac Lab）

```bash
python scripts/load_in_isaaclab.py --convert-only
```

或者用 Isaac Lab 官方工具：

```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    <绝对路径>/urdf/tiangong3_isaaclab.urdf \
    <绝对路径>/usd \
    --joint-stiffness 200.0 --joint-damping 20.0
```

> ✅ **已在实机核对（Isaac Lab v2.3.2 + Isaac Sim 5.1.0）**。网上流传的"这些参数已废弃"
> 说法对**这个版本是错的**，以 `source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py`
> 源码为准：
>
> - `usd_file_name: str | None = None` —— **仍然存在**。为 `None` 时自动取 URDF 文件名的词干：
>   `tiangong3_isaaclab.urdf` → `tiangong3_isaaclab.usd`
> - `collider_type: Literal["convex_hull", "convex_decomposition"] = "convex_hull"` —— **存在**
> - `make_instanceable: bool = True` —— **存在**（定义在基类 `AssetConverterBaseCfg` 上）
> - `fix_base` 是**必填**项（无默认值），人形机器人必须显式写 `False`，否则被钉在空中
> - `merge_fixed_joints` 默认 `True`，会合并固定关节；要保住 39 个 link 必须写 `False`
> - `joint_drive` 的嵌套是 `JointDriveCfg(drive_type, target_type, gains=PDGainsCfg(stiffness, damping))`
>
> **输出路径**由源码 `usd_path = os.path.join(usd_dir, usd_file_name)` 决定，即
> **平铺**在 `usd/` 下（`usd/tiangong3_isaaclab.usd`），不是 `usd/<name>/<name>.usda` 子目录。

### 第 3 步：加载 + 站立测试

```bash
python scripts/load_in_isaaclab.py --headless --steps 500
```

会打印全部 39 个刚体、31 个关节的名单与限位，然后跑 500 步站立测试。
看最后一行：

- `✓ 机器人保持站立` → URDF 加载正常
- `✗ 机器人倒下了` → PD 增益太小，提高 `stiffness`（200 → 500）再试

---

## 四、其他环境

**MuJoCo**：需要把 URDF 转成 MJCF。用 `mujoco` 自带的转换器：

```bash
python -m mujoco.urdf2mjcf tiangong3_isaaclab.urdf
```

注意 MuJoCo 对凸包的处理方式与 Isaac Lab 不同，可能需要调整碰撞参数。

**ROS 2**：`package://` 路径要在 `package.xml` + `CMakeLists.txt` 里正确注册包名。
建议用 `Open-X-Humanoid/TienKung_ROS` 仓库的现成配置，别自己搭。

**Pinocchio / PyBullet**：直接用 `tiangong3_isaaclab.urdf`，两者都能读相对路径的 mesh。

---

## 五、已知问题与不确定性

诚实地列一下我没能 100% 验证的部分：

1. **凸包外形未做定量评估** —— 我知道它比精细版大，但没测过具体大多少（比如
   pelvis 凸包 vs 真实外形差几厘米）。做**精细抓取/接触任务**前建议自己量一下。

2. **惯性参数未经物理验证** —— 质量总和 65.82 kg 看起来合理，但我没有真机数据
   可比对。如果做 sim-to-real，**务必与真机参数核对**，否则动力学仿真会偏。

3. **`tiangong3_0_V2_urdf` 这个包名含义不明** —— 看名字像 "V2 版本"，
   但仓库里没有 V1 或更新的版本，无法确认是否最新。**建议向天工官方确认版本**。

4. **未在真实 Isaac Lab 环境测试** —— 打包环境是 Windows，而 Isaac Lab 只支持
   Linux。脚本按官方文档的 API 写的，但**首次在 Linux 上跑时请留意报错**。
   文档里的 API 变化信息来自 Isaac Lab 官方迁移指南，可信度较高。

5. **`meshes_convex` 与 `meshes` 的文件名完全对应** —— 我核对过 39 个名前缀一致，
   所以换目录时不需要改名。但如果你只想要部分零件，注意 URDF 引用的是全部 39 个。

---

## 六、出问题了怎么查

| 现象 | 原因 | 解决 |
|---|---|---|
| `Cannot find mesh file` | 用了 `tiangong3.urdf` | 换 `tiangong3_isaaclab.urdf` |
| 转换成功但模型是空的 | 同上，路径解析失败 | 同上 |
| 机器人悬空不动 | `fix_base=True` | 改成 `False` 重转 |
| 按 link 名找不到（如 `pelvis`） | `merge_fixed_joints=True` 合并了 | 改成 `False` |
| 显存溢出 / 训练极慢 | 用了 136 MB 精细网格 | 确认用 `meshes_convex/` |
| 机器人立刻瘫倒 | PD 增益太小 | 提高 `stiffness` |
| 关节乱转 / 方向相反 | 关节名或零位不匹配 | `--print-joints` 打印真实名单核对 |
| 找不到输出的 USD | 以为会生成到 `usd/<name>/` 子目录 | 实际平铺在 `usd/tiangong3_isaaclab.usd` |

更多细节见 `docs/README_detailed_isaaclab.md`（含 URDF 原理的完整讲解）。

---

## 七、许可与出处

- 原始资产版权归 **北京人形机器人创新中心（X-Humanoid）** 所有
- 仓库版权文件见上游 `licence`
- 本包仅为路径修正 + 格式整理的再分发，**未修改任何几何或动力学数据**
- 商用前请自行确认上游许可条款
