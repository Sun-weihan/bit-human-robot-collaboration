#!/usr/bin/env python3
"""提高手臂 PD 刚度（只改 stiffness/damping，不碰力矩/速度上限）。

教训：上一次用「整个 actuator block 内替换所有同名键」的正则，
把 effort_limit_sim(肩 90 N·m) 和 velocity_limit_sim(7.22 rad/s) 也改成了 400，
等于把手臂力矩上限放大 4 倍——真机上会烧电机。
这次只在 stiffness={...} 和 damping={...} 两个子块内部替换。
"""
import io
import re

ASSET = "/root/autodl-tmp/TienKung-Lab/legged_lab/assets/tiangong3/tiangong3.py"

TARGET = {
    "shoulder_pitch": (400.0, 12.0),
    "shoulder_roll": (300.0, 9.0),
    "shoulder_yaw": (300.0, 9.0),
    "elbow_pitch": (400.0, 12.0),
    "elbow_yaw": (300.0, 9.0),
}


def sub_in_brace_block(text, header, mapping):
    """只替换 header={ ... } 花括号内部的值。"""
    i = text.index(header)
    j = text.index("{", i)
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    seg = text[j:k]
    for nm, val in mapping.items():
        seg = re.sub(r'("' + nm + r'_[^"]*"\s*:\s*)[\d.]+',
                     lambda m, v=val: m.group(1) + str(v), seg)
    return text[:j] + seg + text[k:]


def main():
    s = io.open(ASSET, encoding="utf-8").read()

    # 肩 / 肘：stiffness 与 damping 分别是独立的字典块
    for blk, keys in (('"shoulder": ImplicitActuatorCfg(', ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")),
                      ('"elbow": ImplicitActuatorCfg(', ("elbow_pitch", "elbow_yaw"))):
        i = s.index(blk)
        j = s.index("),", i)
        seg = s[i:j]
        # 在 seg 内部找 stiffness={ 和 damping={ 两个子块分别替换
        for hdr, idx in (("stiffness={", 0), ("damping={", 1)):
            if hdr in seg:
                # 用绝对位置替换
                abs_i = i + seg.index(hdr)
                s = sub_in_brace_block(s, hdr if False else s[abs_i:abs_i + len(hdr)],
                                       {k2: TARGET[k2][idx] for k2 in keys})
        # 重新定位（s 变了）
        i = s.index(blk)
        j = s.index("),", i)

    # 腕：标量形式
    i = s.index('"wrist": ImplicitActuatorCfg(')
    j = s.index(")", s.index("damping=", i))
    seg = s[i:j]
    seg = seg.replace("stiffness=50.0", "stiffness=150.0").replace("damping=2.5", "damping=6.0")
    s = s[:i] + seg + s[j:]

    io.open(ASSET, "w", encoding="utf-8").write(s)
    print("  已只改 stiffness/damping")

    # 校验
    t = io.open(ASSET, encoding="utf-8").read()
    for nm in ("shoulder_pitch", "shoulder_roll", "elbow_pitch", "elbow_yaw"):
        lim = re.search(r'effort_limit_sim[^}]*"' + nm + r'_[^"]*"\s*:\s*([\d.]+)', t, re.S)
        print(f"    {nm:<16} effort_limit = {lim.group(1) if lim else '标量'}")


if __name__ == "__main__":
    main()
