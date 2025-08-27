private IntPtr GetDacHandle(bool useCDac)
{
    // ADD THIS: Custom trace listener setup (only once per process)
    const string traceFileName = "/tmp/sos_dac_load.txt";
    const string traceListenerName = "SosDacTraceListener";
    
    // Check if our custom listener is already added
    bool listenerExists = Trace.Listeners
        .Cast<TraceListener>()
        .Any(l => l.Name == traceListenerName);
    
    if (!listenerExists)
    {
        try
        {
            // Create a new listener that writes to our file
            var listener = new TextWriterTraceListener(
                File.Open(traceFileName, FileMode.Append, FileAccess.Write, FileShare.ReadWrite)
            ) { Name = traceListenerName };
            
            // Set formatting for better readability
            listener.TraceOutputOptions = TraceOptions.DateTime | TraceOptions.ThreadId;
            
            // Add to trace listeners
            Trace.Listeners.Add(listener);
            
            // Flush immediately after each write
            Trace.AutoFlush = true;
            
            // Log initialization
            Trace.TraceInformation($"[INIT] SOS DAC tracing initialized. Log file: {traceFileName}");
        }
        catch (Exception ex)
        {
            // Fallback to stderr if file logging fails
            Console.Error.WriteLine($"[SOS] Failed to initialize DAC trace file: {ex.Message}");
        }
    }

    // BEGIN EXISTING CODE (with added trace statements)
    bool verifySignature = false;
    Trace.TraceInformation($"[START] Attempting to get DAC handle (useCDac={useCDac})");
    
    string dacFilePath = useCDac ? _runtime.GetCDacFilePath() : _runtime.GetDacFilePath(out verifySignature);
    
    if (dacFilePath == null)
    {
        string errorMsg = $"Could not find matching DAC {useCDac} for runtime: {_runtime.RuntimeModule.FileName}";
        Trace.TraceError($"[FAIL] {errorMsg}");
        return IntPtr.Zero;
    }
    
    Trace.TraceInformation($"[FOUND] DAC file path: {dacFilePath}, verifySignature={verifySignature}");

    IntPtr dacHandle = IntPtr.Zero;
    IDisposable fileLock = null;
    try
    {
        if (verifySignature)
        {
            Trace.TraceInformation($"[VERIFY] Verifying DAC signing and cert for {dacFilePath}");
            
            // Check if the DAC cert is valid before loading
            if (!AuthenticodeUtil.VerifyDacDll(dacFilePath, out fileLock))
            {
                Trace.TraceError($"[VERIFY] DAC signature verification failed for {dacFilePath}");
                return IntPtr.Zero;
            }
            Trace.TraceInformation($"[VERIFY] DAC signature verification succeeded for {dacFilePath}");
        }
        
        try
        {
            Trace.TraceInformation($"[LOAD] Attempting to load DAC library: {dacFilePath}");
            dacHandle = DataTarget.PlatformFunctions.LoadLibrary(dacFilePath);
            
            if (dacHandle == IntPtr.Zero)
            {
                Trace.TraceError($"[LOAD] LoadLibrary returned NULL for {dacFilePath}");
                return IntPtr.Zero;
            }
            
            Trace.TraceInformation($"[LOAD] Successfully loaded DAC at handle: {dacHandle}");
        }
        catch (Exception ex) when (ex is DllNotFoundException or BadImageFormatException)
        {
            Trace.TraceError($"[LOAD] LoadLibrary({dacFilePath}) FAILED: {ex.GetType().Name} - {ex.Message}");
            return IntPtr.Zero;
        }
    }
    finally
    {
        // Keep DAC file locked until it loaded
        fileLock?.Dispose();
        Trace.TraceInformation($"[LOCK] DAC file lock released");
    }
    
    Debug.Assert(dacHandle != IntPtr.Zero);
    Trace.TraceInformation($"[ASSERT] DAC handle validation: {(dacHandle == IntPtr.Zero ? "FAILED" : "SUCCESS")}");
    
    if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
    {
        Trace.TraceInformation($"[UNIX] Invoking DllMain for non-Windows platform");
        try
        {
            DllMainDelegate dllmain = SOSHost.GetDelegateFunction<DllMainDelegate>(dacHandle, "DllMain");
            if (dllmain != null)
            {
                dllmain.Invoke(dacHandle, 1, IntPtr.Zero);
                Trace.TraceInformation($"[UNIX] DllMain invoked successfully");
            }
            else
            {
                Trace.TraceError($"[UNIX] Failed to get DllMain function pointer");
            }
        }
        catch (Exception ex)
        {
            Trace.TraceError($"[UNIX] Error invoking DllMain: {ex.Message}");
        }
    }
    
    Trace.TraceInformation($"[COMPLETE] DAC handle obtained: {dacHandle}");
    return dacHandle;
}
