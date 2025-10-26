import lldb
import os
import sys

def main():
    lldb.SBDebugger.Initialize()
    dbg = lldb.SBDebugger.Create()
    dbg.SetAsync(False)

    # Get the executable path. Use os.path.expanduser to resolve '~'.
    exe_path = os.path.expanduser("~/dbgcopilot/examples/crash_demo/crash")

    # Check if the executable exists
    if not os.path.exists(exe_path):
        print(f"Error: Executable '{exe_path}' not found.")
        lldb.SBDebugger.Destroy(dbg)
        sys.exit(1)

    # Create target
    err = lldb.SBError()
    # Specify host platform name to prefer the local process plugin if necessary.
    # This might help in some ambiguous LLDB configurations.
    target = dbg.CreateTarget(exe_path, None, "host", True, err)
    if not target or not target.IsValid():
        # Fallback without specifying platform name
        target = dbg.CreateTargetWithFileAndArch(exe_path, None)
        if not target or not target.IsValid():
            raise RuntimeError(f"Failed to create target: {err}")
    
    print("Target created successfully.")

    # Launch the process using LaunchSimple()
    print("Attempting to launch process with LaunchSimple()...")
    process = target.LaunchSimple([], None, os.getcwd())

    if process and process.IsValid():
        # Get the process state and convert it to a string.
        state_enum = process.GetState()
        state_str = dbg.StateAsCString(state_enum)
        
        # Check if the process actually launched (i.e., state is not just eStateExited immediately)
        if state_enum != lldb.eStateExited:
            print("Process launched successfully!")
            print(f"PID: {process.GetProcessID()}")
            print(f"State: {state_str}")

            # Wait for the process to terminate.
            print("Waiting for process to exit...")
            while process.GetState() != lldb.eStateExited:
                pass
            
            exit_status = process.GetExitStatus()
            print(f"Process exited with status: {exit_status}")
        else:
            print("Failed to launch the process; it exited immediately.")
            print(f"Process exit status: {process.GetExitStatus()}")
    else:
        print("Failed to launch the process. The returned process object is invalid.")


    lldb.SBDebugger.Destroy(dbg)

if __name__ == "__main__":
    main()


