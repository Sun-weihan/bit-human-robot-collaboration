"""bit-human-robot-collaboration

人机协同科研项目 —— 环境自检与最小可运行示例。

运行方式:
    python main.py

预期输出:
    打印环境信息，并执行一次最小协同任务模拟。
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# 环境自检
# ---------------------------------------------------------------------------
def check_environment() -> dict[str, str]:
    """采集当前解释器与系统环境信息，用于确认仓库可正常运行。"""
    return {
        "python_version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }


# ---------------------------------------------------------------------------
# 最小协同任务模型
# ---------------------------------------------------------------------------
@dataclass
class Collaborator:
    """一个参与协同的主体（人类或机器人）。"""

    name: str
    role: str  # "human" 或 "robot"
    skills: list[str] = field(default_factory=list)

    def can_handle(self, task: str) -> bool:
        return task in self.skills


@dataclass
class Task:
    """待分配的协同任务。"""

    name: str
    required_skill: str
    assignee: str | None = None
    status: str = "pending"


def assign(tasks: list[Task], collaborators: list[Collaborator]) -> list[Task]:
    """按技能匹配把任务分配给协同主体。

    匹配规则：拥有所需技能的主体中，按列表顺序取第一个可用的。
    """
    for task in tasks:
        for person in collaborators:
            if person.can_handle(task.required_skill):
                task.assignee = person.name
                task.status = "assigned"
                break
        else:
            task.status = "unassigned"
    return tasks


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("bit-human-robot-collaboration :: 环境自检")
    print("=" * 60)

    env = check_environment()
    for key, value in env.items():
        print(f"  {key:<18} : {value}")

    print()
    print(f"  当前时间(UTC)      : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")

    print()
    print("=" * 60)
    print("最小协同任务分配示例")
    print("=" * 60)

    collaborators = [
        Collaborator("Operator-A", "human", skills=["perception", "decision"]),
        Collaborator("Arm-01", "robot", skills=["grasp", "decision"]),
        Collaborator("Arm-02", "robot", skills=["grasp"]),
    ]

    tasks = [
        Task("识别目标物体", "perception"),
        Task("抓取目标物体", "grasp"),
        Task("规划抓取路径", "decision"),
        Task("焊接电路板", "welding"),  # 无人具备该技能
    ]

    for task in assign(tasks, collaborators):
        mark = "OK " if task.status == "assigned" else "!! "
        print(f"  {mark}{task.name:<12} -> {task.assignee or '(无匹配主体)'}")

    assigned = sum(1 for t in tasks if t.status == "assigned")
    print()
    print(f"  分配成功 {assigned}/{len(tasks)} 个任务。")

    if assigned == len(tasks):
        print("  结论：协同分配逻辑正常。")
    else:
        print("  注意：存在未分配任务，属预期行为（用于验证兜底分支）。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
