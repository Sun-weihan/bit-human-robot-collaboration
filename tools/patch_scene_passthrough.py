#!/usr/bin/env python3
"""让仓库的 SceneCfg 把 env 配置里声明的额外刚体透传到场景。

问题：legged_lab/utils/env_utils/scene.py 的 SceneCfg.__init__ 是硬编码的——
它只用 config 的 num_envs / env_spacing / terrain_* / robot 等已知字段，
逐个手工构建 terrain、robot、contact_sensor、light、sky_light。
所以在 cfg.scene 上新增的 carry_box（无论动态赋值还是声明成字段）都会被丢掉。

修复：在 SceneCfg.__init__ 末尾加一段通用透传——
把 config 上所有 RigidObjectCfg 类型的字段（robot 除外）复制进场景，
并把 prim_path 里的 {ENV_REGEX_NS} 填好。这样以后加任何刚体都不用再改这个文件。
"""
import io

P = "/root/autodl-tmp/TienKung-Lab/legged_lab/utils/env_utils/scene.py"

MARK = "        # ---- 额外刚体透传"

PATCH = '''
        # ---- 额外刚体透传 ----
        # env 配置里声明的 RigidObjectCfg（例如搬运箱）在这里复制进场景。
        # 本方法原本只手工构建 terrain/robot/contact_sensor/light/sky_light，
        # 不读 config 上的其他字段，导致自定义刚体永远进不了场景。
        try:
            from isaaclab.assets import RigidObjectCfg as _RigidCfg

            for _name, _val in list(vars(config).items()):
                if _name in ("robot", "terrain"):
                    continue
                if isinstance(_val, _RigidCfg):
                    _cfg = _val.replace(
                        prim_path="{ENV_REGEX_NS}/" + _name
                    ) if hasattr(_val, "replace") else _val
                    setattr(self, _name, _cfg)
        except Exception as _e:  # noqa: BLE001
            print(f"[SceneCfg] 额外刚体透传失败: {_e}")
'''


def main():
    s = io.open(P, encoding="utf-8").read()
    if MARK in s:
        print("  已有透传代码，跳过")
        return

    # 插到 __init__ 的末尾：找 __init__ 里最后一个 self.xxx 赋值之后不好定位，
    # 改用「文件里第一个 def __init__ 之后、下一个 def 之前」的方式
    lines = s.splitlines()
    init_start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("def __init__(self, config"):
            init_start = i
            break
    if init_start is None:
        raise SystemExit("没找到 SceneCfg.__init__")

    # 找下一个同级 def
    init_end = None
    for j in range(init_start + 1, len(lines)):
        ln = lines[j]
        if ln.strip().startswith("def ") and (len(ln) - len(ln.lstrip())) <= 4:
            init_end = j
            break
    if init_end is None:
        init_end = len(lines)

    lines = lines[:init_end] + PATCH.rstrip("\n").splitlines() + lines[init_end:]
    io.open(P, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"  已给 {P} 加上刚体透传（插在第 {init_end} 行前）")


if __name__ == "__main__":
    main()
