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
```

> 注意：入口脚本 `main.py` 已在提交 `2c7b505 delete test file` 中被移除，当前尚未恢复，
> 因此克隆后暂时没有可直接运行的脚本。

## 目录结构

```
.
├── README.md                          # 项目说明
├── .gitignore                         # Git 忽略规则
├── .vscode/                           # 编辑器推荐配置
│   ├── settings.json
│   └── extensions.json
├── 2027 AI_机器人顶会投稿时间轴.docx   # 学习资料：投稿时间轴
└── git相关操作.md                      # 学习资料：Git 常用操作笔记
```

## 说明

本仓库处于初始化阶段。此前 `main.py` 提供环境自检与最小协同任务分配逻辑骨架
（`Collaborator` / `Task` / `assign`），恢复后可作为验证仓库可运行性的入口。
