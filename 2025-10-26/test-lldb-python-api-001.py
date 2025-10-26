import lldb, os

lldb.SBDebugger.Initialize()
dbg = lldb.SBDebugger.Create()
dbg.SetAsync(False)
interp = dbg.GetCommandInterpreter()

def do(cmd):
    res = lldb.SBCommandReturnObject()
    interp.HandleCommand(cmd, res)
    print(res.GetOutput() or res.GetError() or "")
    return res.Succeeded()

# 1) Force host platform
do("platform select host")
do("platform status")

# 2) Best-effort: disable LLGS for local if this build supports it
# do("settings set platform.plugin.linux.use-llgs-for-local false")

# 3) Create target and launch
err = lldb.SBError()
target = dbg.CreateTarget("~/dbgcopilot/examples/crash_demo/crash", None, "host", False, err)
if not target or not target.IsValid():
    raise RuntimeError("Failed to create target")

# Get the host platform and explicitly set it for the target
#platform = dbg.GetSelectedPlatform()
#if platform and platform.IsValid():
#    target.SetPlatform(platform)

launch = lldb.SBLaunchInfo([])
#err = lldb.SBError()
process = target.Launch(launch, err)
print("launch error:", err)
