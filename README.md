# bit-human-robot-collaboration

人机协同科研项目工作仓库。

## 环境要求

- Python 3.10+（推荐 3.12/3.13）
- 无第三方依赖，仅使用标准库

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/Sun-weihan/bit-human-robot-collaboration.git
cd bit-human-robot-collaboration

# 运行自检脚本
python main.py
```

预期输出：打印当前 Python 与系统环境信息，并完成一次最小协同任务分配模拟。

## 目录结构

```
.
├── main.py          # 环境自检 + 最小协同任务示例
├── README.md        # 项目说明
└── .gitignore       # Git 忽略规则
```

## 说明

本仓库处于初始化阶段。`main.py` 用于验证仓库可正常克隆、运行，
同时提供一个可扩展的协同任务分配逻辑骨架（`Collaborator` / `Task` / `assign`）。
