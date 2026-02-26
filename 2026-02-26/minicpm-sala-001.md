根据你提供的两个官方文档，我为你整理了从申请FCloud实例到配置比赛环境的完整操作步骤。这将是我们备赛的基础设施搭建指南。

### **第一步：申请并配置FCloud GPU实例**
根据 [FCloud Omnibot 平台使用指南](https://docs.cnomnibot.com/zh/use/document.html)，这是获取比赛指定硬件（RTX PRO 6000D）的途径。

1.  **注册账号**：访问 [FCloud控制台注册页](https://console.cnomnibot.com/login)，使用手机号或邮箱完成账号注册。
2.  **实名认证**：登录后，在控制台找到实名认证页面，根据指引完成个人或企业认证。这是使用国内云服务的必要步骤。
3.  **账户充值**：进入“账户概览”为账户充值。平台支持支付宝、微信支付，按分钟计费（不足1分钟按1分钟计），方便灵活控制成本。
4.  **创建训练任务（获取实例）**：
    *   从控制台选择 **“具身智能服务”**。
    *   点击左侧菜单 **“训练服务”**，**区域务必选择“上海”**（这是文档指定的区域），点击确认。
    *   点击右上角 **“创建任务”**。
    *   在硬件配置中，**选择 `RTX PRO 6000D`** 并按需选择显卡数量。系统会实时显示价格，确认后即可启动实例。

### **第二步：在实例上搭建比赛环境**
实例启动后，你需要连接并进行配置。根据 [比赛官方工具包](https://soar.openbmb.cn/toolkit) 的指引，最便捷的方式是使用他们提供的基础镜像。

1.  **连接实例**：通过SSH连接到刚创建的FCloud实例。
2.  **拉取官方基础镜像**：在实例中执行以下命令，拉取已内置SGLang框架和必要依赖的Docker镜像（推荐使用国内阿里云镜像加速）：
    ```bash
    docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest
    ```
3.  **下载模型文件**：在实例中，使用Hugging Face或ModelScope CLI工具将比赛指定的模型 `OpenBMB/MiniCPM-SALA` 下载到本地目录（例如 `~/models`）。
    ```bash
    # 使用 ModelScope 示例
    pip install modelscope
    modelscope download --model OpenBMB/MiniCPM-SALA --local_dir ./models
    ```
4.  **启动模型服务容器**：使用下载好的镜像和模型，启动SGLang推理服务。注意要挂载模型目录并映射端口。
    ```bash
    docker run -d \
      --name soar-sglang-server \
      --gpus 'device=0' \
      -p 30000:30000 \
      -v ~/models/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
      modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest
    ```
    至此，模型服务已在后台运行，可以通过本地的30000端口访问。

### **第三步：验证环境与后续优化**
环境启动后，必须验证其正确性并熟悉评测工具。

1.  **验证模型正确性**：使用官方提供的 `eval_model.py` 脚本，对公开数据集 `perf_public_set.jsonl` 进行测试，确保服务运行正常且模型输出正确。
    ```bash
    # 需要在容器外或另一终端执行，确保能访问到服务的30000端口
    python3 eval_model.py \
      --api_base http://127.0.0.1:30000 \
      --model_path /path/to/your/models \ # 指向你下载的模型文件
      --data_path /path/to/perf_public_set.jsonl \
      --concurrency 32
    ```
2.  **理解评测流程**：仔细阅读工具包文档中关于 **“模型速度评测”** (`bench_serving.sh`) 和 **“构建注意事项”** 的部分。特别注意，当你开始编写自定义CUDA算子进行优化时，最终的代码提交（wheel包）必须严格遵守其文件结构要求，**只包含 `sgl_kernel/` 相关文件**，否则将导致提交失败。

### **给你的执行建议**
*   **成本控制**：FCloud按分钟计费非常灵活。建议开发调试阶段（如第一步、第二步）使用按需实例，结束后及时释放。需要长时间跑基准测试或训练时再启动。
*   **网络注意**：由于我们在中国，FCloud选择“上海”区域，以及使用阿里云镜像拉取Docker镜像，都是为了获得更快的网络速度。
*   **求助点**：如果在FCloud账户注册、充值或创建训练任务时遇到问题，文档最后提到“可通过企业客服与我们联系”。我可以帮你草拟询问邮件。

现在，你可以先从第一步开始操作。如果在FCloud注册或创建实例过程中遇到任何界面或流程问题，随时告诉我，我们一起解决。
