<#
.SYNOPSIS
    把天工 3.0 URDF 资源包部署到 AutoDL 上的 IsaacLab 实例，并在服务器上跑通。

.DESCRIPTION
    做两件事：
      1. 用 scp 把本地 tiangong3/ 资源包同步到服务器；
      2. 通过 ssh 执行服务器端引导脚本 deploy/remote_bootstrap.sh。

    引导脚本的模式：
      report   只体检（GPU / 磁盘 / conda / IsaacLab 版本），不碰仿真，几秒钟
      convert  URDF -> USD 转换（首次 1~3 分钟）
      test     载入场景 + 资产完整性核对 + 重力沉降测试
      all      report + convert + test

.NOTES
    为什么必须走这个脚本、不能直接 python 跑：
      * IsaacLab 的 isaaclab.sh 从 $CONDA_PREFIX 取 python，所以脚本里先 conda activate；
      * py 脚本内部必须先启动 AppLauncher 再导入转换器（否则 pxr 不存在）；
      * 必须 PYTHONUNBUFFERED=1，否则 simulation_app.close() 会吞掉末尾输出。
    这些都已写进 remote_bootstrap.sh 和 load_in_isaaclab.py，不用你手动记。

.EXAMPLE
    .\deploy.ps1 -Mode report
    .\deploy.ps1 -Mode all
    .\deploy.ps1 -Mode test -SkipUpload
#>
[CmdletBinding()]
param(
    [string] $Server    = 'root@connect.bjb1.seetacloud.com',
    [int]    $Port      = 56456,
    [string] $Key       = "$env:USERPROFILE\.ssh\id_ed25519",
    [string] $RemoteDir = '/root/autodl-tmp/tiangong3',

    # 本地资源包目录（默认 = 本脚本上一级的 tiangong3/）
    [string] $LocalPkg  = (Join-Path (Split-Path -Parent $PSScriptRoot) 'tiangong3'),

    [ValidateSet('report', 'convert', 'test', 'all')]
    [string] $Mode = 'all',

    # 只跑远程，不重新上传
    [switch] $SkipUpload
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 1. 前置检查
# ---------------------------------------------------------------------------
if (-not (Test-Path $Key)) { throw "找不到 SSH 私钥: $Key" }

if (-not $SkipUpload) {
    if (-not (Test-Path $LocalPkg)) { throw "找不到本地资源包: $LocalPkg" }
    foreach ($need in @('urdf', 'meshes_convex', 'scripts')) {
        if (-not (Test-Path (Join-Path $LocalPkg $need))) {
            throw "资源包不完整，缺少 $need/：$LocalPkg"
        }
    }
}

$sshExe = (Get-Command ssh -ErrorAction Stop).Source
$scpExe = (Get-Command scp -ErrorAction Stop).Source

# 已知主机记录写到工作区里，不去动用户全局的 ~/.ssh/known_hosts
$knownHosts = Join-Path $PSScriptRoot 'known_hosts'

$commonArgs = @(
    '-i', $Key
    '-o', 'BatchMode=yes'
    '-o', 'StrictHostKeyChecking=accept-new'
    '-o', "UserKnownHostsFile=$knownHosts"
    '-o', 'ConnectTimeout=30'
)

# ---------------------------------------------------------------------------
# 2. 上传
# ---------------------------------------------------------------------------
if (-not $SkipUpload) {
    Write-Host "==> 上传资源包 -> ${Server}:${RemoteDir}" -ForegroundColor Cyan
    # 确保远端目录存在
    & $sshExe @commonArgs -p $Port $Server "mkdir -p $RemoteDir"
    if ($LASTEXITCODE -ne 0) { throw "远端 mkdir 失败" }

    # 注意 -r 后面跟的是本地目录，scp 会把目录本身放到目标下
    & $scpExe @commonArgs -r -P $Port $LocalPkg "${Server}:$(Split-Path -Parent $RemoteDir)/"
    if ($LASTEXITCODE -ne 0) { throw "scp 上传失败（退出码 $LASTEXITCODE）" }
    Write-Host "    上传完成" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 3. 远程执行
# ---------------------------------------------------------------------------
$remoteScript = "$RemoteDir/deploy/remote_bootstrap.sh"
Write-Host "==> 远程执行 [$Mode] : $remoteScript" -ForegroundColor Cyan

# Isaac Sim 会在 stderr 上刷一堆无害警告，这里过滤掉常见的几类，便于看结果
$noise = 'GLFW initialization failed|carb.windowing-glfw|Enable omni.materialx|' +
         'dynamic_control is deprecated|non optional plugin interface|' +
         'CPU performance profile|No Viewport Window|failed to open the default display'

$remote = "bash $remoteScript $RemoteDir $Mode 2>&1 | grep -Ev '$noise'"
& $sshExe @commonArgs -p $Port $Server $remote
$rc = $LASTEXITCODE

Write-Host ""
if ($rc -eq 0) {
    Write-Host "==> 完成（退出码 0）" -ForegroundColor Green
} else {
    Write-Host "==> 失败（退出码 $rc）" -ForegroundColor Red
    Write-Host "    完整日志在服务器: /tmp/tk3_*.log" -ForegroundColor Yellow
}
exit $rc
