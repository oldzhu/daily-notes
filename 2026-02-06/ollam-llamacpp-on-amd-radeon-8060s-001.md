set path=%path%;C:\Users\orien\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\_rocm_sdk_libraries_custom\bin

ollama
....
time=2026-02-05T19:41:13.707+08:00 level=INFO source=server.go:430 msg="starting runner" cmd="C:\\Users\\orien\\AppData\\Local\\go-build\\67\\6773b803f0edcaea82ffc897421089eede2db9647af13a67df1c1c469a46ec5f-d\\ollama.exe runner --ollama-engine --port 59700"
time=2026-02-05T19:41:14.032+08:00 level=INFO source=server.go:430 msg="starting runner" cmd="C:\\Users\\orien\\AppData\\Local\\go-build\\67\\6773b803f0edcaea82ffc897421089eede2db9647af13a67df1c1c469a46ec5f-d\\ollama.exe runner --ollama-engine --port 59706"
time=2026-02-05T19:41:15.595+08:00 level=INFO source=types.go:42 msg="inference compute" id=0 filter_id=0 library=ROCm compute=gfx1151 name=ROCm0 description="AMD Radeon(TM) 8060S Graphics" libdirs=ollama driver=70151.80 pci_id=0000:c7:00.0 type=iGPU total="96.0 GiB" available="94.1 GiB"
time=2026-02-05T19:41:15.595+08:00 level=INFO source=routes.go:1725 msg="vram-based default context" total_vram="96.0 GiB" default_num_ctx=262144
...
llama.cpp
....

C:\localrepos\llama.cpp\build\bin>llama-cli
HIP Library Path: C:\WINDOWS\SYSTEM32\amdhip64_7.dll
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon(TM) 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
error: --model is required
...

