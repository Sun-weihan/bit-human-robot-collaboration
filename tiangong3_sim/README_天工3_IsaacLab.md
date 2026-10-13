# 天工 3.0 URDF 用于 Isaac Lab 仿真

> 资材来源：https://github.com/Open-X-Humanoid/TienKung_URDF （目录 `tiangong3_urdf/`）
> 整理日期：2026-10-13

---

## 一、我用什么拿到了什么

| 产出 | 路径 | 说明 |
|---|---|---|
| 原始 URDF | `urdf/tiangong3.urdf` | 从官方仓库原样下载，42 KB / 1120 行 |
| **Isaac Lab 版 URDF** | `urdf/tiangong3_isaaclab.urdf` | **路径已修正，直接可用** |
| 凸包网格 | `meshes_convex/*.STL` | 39 个，合计 3.23 MB |
| 获取脚本 | `scripts/fetch_assets.py` | 可重复运行，支持断点续传 |
| 加载脚本 | `scripts/load_in_isaaclab.py` | URDF→USD 转换 + 场景搭建 |

**实测验证结果**：

```
link          : 39        <- 每个都带 <inertial>
joint         : 38  (revolute 31, fixed 7)
inertial      : 39        <- 有惯性才能做动力学仿真
总质量        : 65.82 kg
root link     : pelvis
mesh 引用     : 78 处，全部指向存在的文件 ✓
STL 完整性    : 39/39 合法二进制，共 67,574 三角面
```

---

## 二、URDF 到底是干什么的

**一句话**：URDF 是机器人的「结构说明书」——它只描述**机器人长什么样**，
不描述**机器人怎么动**。

具体来说，一个 URDF 文件定义了这四件事。这是理解它的核心框架：

| 概念 | 英文 | 作用 | 类比 |
|---|---|---|---|
| **连杆** | `<link>` | 一个刚体（大腿、小腿、躯干） | 积木块 |
| **关节** | `<joint>` | 连接两个连杆，定义如何相对运动 | 积木之间的轴 |
| **可视化** | `<visual>` | 长什么样（用于渲染） | 外观贴图 |
| **碰撞体** | `<collision>` | 撞到什么（用于物理计算） | 隐形的简化外壳 |

再加两个关键附加属性：

- `<inertial>` —— **质量与转动惯量**。没有它，物理引擎无法计算动力学，
  机器人会在仿真里"飘"起来或直接报错。
- `<limit>` —— **关节限位与力矩上限**。这就是为什么真实机器人膝盖不能反折。

天工 3.0 的 URDF 里，这五样是全的：

```
39 个 link，每个都有 <inertial>（质量 + 惯量矩阵）
38 个 joint，31 个 revolute（可动）+ 7 个 fixed（固定）
每个 revolute 都带 <limit>（lower / upper / effort / velocity）
```

---

## 三、为什么 Isaac Lab 需要它

这是最关键的一环，理解了这个就理解了整条链路：

```
   URDF                →        USD              →        Isaac Lab
（结构说明书）            （NVIDIA 的场景格式）      （GPU 并行仿真）

  人类可读的 XML            二进制优化格式              物理引擎 + 训练框架
  与 ROS 生态兼容           支持大规模实例化            支持 GPU 上跑几千个环境
```

### 三个层次的作用

**层次一：几何来源。**
Isaac Lab 里所有物体都是 USD。URDF 是机器人形状的**唯一来源**——
没有它，你只能手搓一个立方体。

**层次二：动力学参数来源。**
质量、惯量、关节限位、力矩上限——这些全部从 URDF 读进来。
**这些参数错了，仿真结果就是错的**，而且往往是"看起来能跑但结果不对"的隐性疾病。
这就是为什么下面要专门讲凸包的事。

**层次三：跨平台一致性。**
天工 3.0 在 ROS 2 里用同一个 URDF 跑真机（见 `TienKung_ROS` 仓库）。
你的仿真 URDF 与真机 URDF 一致，才谈得上 **sim-to-real**——
这正是你学习计划第 13 周的核心目标。

---

## 四、必须处理的三个坑（官方仓库的原始文件不能直接用）

### 坑 1：mesh 路径指向了不存在的包名【已修复】

**问题**：URDF 里写的是

```xml
<mesh filename="package://tiangong3_0_V2_urdf/meshes/pelvis.STL"/>
```

但实际仓库目录叫 `tiangong3_urdf`，且没装 ROS，`package://` 协议无法解析。

**后果**：Isaac Lab 转换时报 `Cannot find mesh file`，或更糟——
**转换成功但模型是空的**，你在视口里什么也看不到。

**我的修复**：全部 78 处替换为相对路径

```xml
<mesh filename="../meshes_convex/pelvis.STL"/>
```

`tiangong3_isaaclab.urdf` 就是修复后的版本。**用这个文件，不要用原始的。**

---

### 坑 2：正确使用凸包而非原始网格【关键】

官方提供了两套网格。数据对比：

| 目录 | 文件数 | 总大小 | 用途 |
|---|---|---|---|
| `meshes/` | 39 | **136.0 MB** | 视觉渲染（精细外观） |
| `meshes_convex/` | 39 | **3.23 MB** | **碰撞计算（凸包简化）** |

**差 42 倍。**

**为什么这很重要**：Isaac Lab 的卖点是**在 GPU 上同时跑几千个环境**。
如果每个环境加载 136 MB 的精细网格：

- 显存直接被撑爆（你的 RTX 4060 只有 8 GB）
- 物理碰撞检测会算到天荒地老（精细网格有几十万三角面）

凸包（convex hull）是包裹网格的"最小凸多面体"，667 万个三角面降到 6.7 万。
碰撞检测只需要判断"有没有相交"，用凸包又快又准。

**我的选择**：默认全部使用 `meshes_convex/`。

> 进阶技巧：Isaac Lab 支持"视觉用精细网格、碰撞用凸包"的方案。
> 如果你要做视觉策略（第 10-12 周 Diffusion Policy），
> 再把 `meshes/` 也下载下来作为 visual 层。当前阶段不需要。

---

### 坑 3：基座必须浮动【已在脚本中处理】

天工 3.0 的 URDF 里**没有** `floating` 关节——root link 是 `pelvis`。

Isaac Lab 转换时必须显式指定 `fix_base=False`。
否则机器人会被**钉死在空中**，你看到的是一具悬空的雕像，
所有训练都不会有反应。这个错误的表现很迷惑人——模型显示正常，就是不动。

---

## 五、完整使用流程

### 第 1 步：获取资材

```bash
cd tiangong3_sim
python scripts/fetch_assets.py
```

脚本会自动处理代理/直连切换，支持重跑续传。

### 第 2 步：转换 URDF → USD

在**装好 Isaac Lab 的 Linux 环境**里：

```bash
# 方式一：用我写的脚本（推荐，会自动配置参数）
python scripts/load_in_isaaclab.py --convert-only

# 方式二：用 Isaac Lab 官方工具
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    <路径>/tiangong3_sim/urdf/tiangong3_isaaclab.urdf \
    <路径>/tiangong3_sim/usd \
    --joint-stiffness 200.0 --joint-damping 20.0
```

> ✅ **已在实机核对（Isaac Lab v2.3.2 + Isaac Sim 5.1.0）**：`usd_file_name` 并未废弃，
> 它仍是 `AssetConverterBaseCfg` 的字段（`str | None = None`）。为 `None` 时自动取 URDF
> 文件名词干，输出 `usd/tiangong3_isaaclab.usd` —— **平铺**在 `usd_dir` 下，
> 而不是 `{usd_dir}/{robot_name}/{robot_name}.usda` 这种子目录结构。
> 同理 `collider_type`、`make_instanceable` 也都存在。详见包根目录 `README.md` 的对照表。

### 第 3 步：在 Isaac Lab 里加载

```bash
python scripts/load_in_isaaclab.py --headless --steps 500
```

脚本会打印全部 39 个刚体、31 个关节的名单与限位，
然后做一次 500 步的站立测试。这是**确认 URDF 正确的第一道关卡**。

---

## 六、你的本机情况说明（重要）

| 项 | 状态 |
|---|---|
| 本机系统 | **Windows 11** |
| 显卡 | RTX 4060 Laptop，8 GB 显存 |
| Isaac Lab 支持 | **仅 Linux** ❌ |

**结论：本机跑不了 Isaac Lab。** 有三个选择：

| 方案 | 说明 | 推荐度 |
|---|---|---|
| **WSL2 + Ubuntu** | 本机就能跑，但 WSL2 的 GPU 直通对 Isaac Sim 支持有限，8 GB 显存偏紧 | ★★ |
| **服务器** | 你的学习计划本来就是"服务器跑重计算"，这是**正路** | ★★★ |
| **只在本机做 URDF 校验** | 用 `urdfpy` / `pinocchio` 做几何与动力学检查 | ★★ |

**建议**：URDF 的准备工作已经在本机完成（下载、路径修正、结构校验）。
剩下的是把它传到服务器，在那边做转换与仿真。
这与你计划里"第 1 周就把服务器环境跑通"的要求一致。

**本机可以立即做的验证**（不需要 Isaac Lab）：

```bash
# 用 pinocchio 做正运动学检查，确认 URDF 能被解析
pip install pin
python -c "
import pinocchio as pin
m = pin.buildModelFromUrdf('urdf/tiangong3_isaaclab.urdf')
print('nq =', m.nq, ' nv =', m.nv)
print('关节数:', m.njoints)
print('总质量:', sum(i.mass for i in m.inertias))
"
```

---

## 七、与学习计划的对接

这份 URDF 正好接上你计划里的几个节点：

| 学习计划 | 对应关系 |
|---|---|
| **第 4-5 周**（状态空间、卡尔曼） | 天工 3.0 是 31 自由度系统，可用它做**多自由度**的 LQR 练习（升级版倒立摆） |
| **第 6-7 周**（运动学、动力学） | URDF 直接喂给 Pinocchio，做题 6.2、7.3 的 RNEA 与重力补偿 |
| **第 13 周**（Isaac Lab RL） | `scripts/load_in_isaaclab.py` 就是起点，后续要改成 manager-based 环境 |
| **第 14 周**（部署） | `TienKung_ROS` 仓库用同一个 URDF 跑真机，是 sim-to-real 的对照基准 |

---

## 八、故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 转换报 `Cannot find mesh` | 用了原始 `tiangong3.urdf` | 改用 `tiangong3_isaaclab.urdf` |
| 模型加载后悬空不动 | `fix_base=True` | 改成 `False`，重转 |
| 显存溢出 / 训练极慢 | 用了 `meshes/` 精细网格 | 确认用 `meshes_convex/` |
| 机器人立刻瘫倒 | PD 增益太小 | 提高 `stiffness`（200→500），或检查 `init_state.joint_pos` |
| 关节乱转 | 关节名不匹配 | 用 `--print-joints` 打印真实名单核对 |
| USD 输出找不到 | 按旧文档指定了 `usd_file_name` | Isaac Lab 3.0 自动命名，去 `usd/<name>/` 找 |

---

## 九、下一步建议

1. **先把 URDF 传上服务器**（本机只做校验）
2. 在服务器跑通 `--convert-only`，确认 USD 生成
3. 跑站立测试，确认 39 个刚体、31 个关节都识别正确
4. 然后进入第 13 周的正式内容：搭 manager-based 环境、写奖励函数

**关于你给的另外三个仓库**：

| 仓库 | 用途 | 什么时候用 |
|---|---|---|
| `TienKung-Lab` | Isaac Lab 官方训练框架 | **第 13 周**——这是你的主线工具 |
| `Deploy_Tienkung` | MuJoCo 部署仿真 | **第 14 周**——部署链路验证 |
| `TienKung_ROS` | ROS 2 集成 | **第 14 周**——部署到真机 |

现在只需要 URDF。但**建议先扫一眼 `TienKung-Lab` 的仓库结构**，
因为第 13 周你大概率要基于它来改，而不是从零搭。
