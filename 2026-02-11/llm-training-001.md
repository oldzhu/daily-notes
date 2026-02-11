# 完整LLM训练管道：C++伪代码实现

我将提供一个从底层到顶层的完整LLM训练管道实现，使用C++风格伪代码展示所有数据结构和算法。这适合你在脑中或纸上模拟整个流程。

## 📁 文件结构概览

```
llm_training_pipeline/
├── 01_data_pipeline/          # 数据处理管道
├── 02_pretraining/           # 预训练阶段
├── 03_sft/                   # 有监督微调
├── 04_reward_modeling/       # 奖励建模
├── 05_rlhf/                  # RLHF阶段
├── 06_alignment/             # 对齐调优
├── 07_evaluation/            # 评估
├── 08_deployment/            # 部署
└── utils/                    # 工具函数
```

## 1. 基础数据结构和类型定义

```cpp
// base_types.hpp
#ifndef BASE_TYPES_HPP
#define BASE_TYPES_HPP

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>
#include <random>
#include <functional>

// ==================== 基础类型定义 ====================
// Basic Type Definitions

// 标量类型
typedef float scalar_t;
typedef int32_t token_id_t;
typedef uint32_t batch_size_t;
typedef uint32_t seq_len_t;
typedef uint32_t vocab_size_t;
typedef uint32_t hidden_dim_t;
typedef uint64_t parameter_count_t;

// 张量基类
class Tensor {
public:
    std::vector<size_t> shape;      // 维度形状
    std::vector<scalar_t> data;     // 数据存储
    bool requires_grad;             // 是否需要梯度
    std::vector<scalar_t> grad;     // 梯度存储
    
    Tensor() : requires_grad(false) {}
    
    Tensor(const std::vector<size_t>& s, bool rg = false) 
        : shape(s), requires_grad(rg) {
        size_t total = 1;
        for (size_t dim : shape) total *= dim;
        data.resize(total, 0.0f);
        if (requires_grad) grad.resize(total, 0.0f);
    }
    
    // 访问元素
    scalar_t& operator[](const std::vector<size_t>& indices) {
        size_t idx = 0;
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[idx];
    }
    
    size_t numel() const {
        size_t total = 1;
        for (size_t dim : shape) total *= dim;
        return total;
    }
};

// 优化器状态
struct OptimizerState {
    scalar_t learning_rate;
    scalar_t beta1;      // Adam beta1
    scalar_t beta2;      // Adam beta2
    scalar_t epsilon;    // Adam epsilon
    int64_t step;        // 当前步数
    
    // 一阶矩和二阶矩
    std::vector<scalar_t> m;  // 一阶矩
    std::vector<scalar_t> v;  // 二阶矩
    
    OptimizerState(scalar_t lr = 1e-3, scalar_t b1 = 0.9, 
                  scalar_t b2 = 0.999, scalar_t eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), 
          epsilon(eps), step(0) {}
};

// 模型配置
struct ModelConfig {
    vocab_size_t vocab_size = 50257;     // 词汇表大小
    hidden_dim_t hidden_size = 768;      // 隐藏层维度
    uint32_t num_layers = 12;            // 层数
    uint32_t num_heads = 12;             // 注意力头数
    uint32_t max_seq_len = 1024;         // 最大序列长度
    uint32_t ffn_dim = 3072;             // FFN维度
    scalar_t dropout_rate = 0.1f;        // Dropout率
    
    // 计算总参数量
    parameter_count_t total_params() const {
        // 简化计算
        parameter_count_t params = 0;
        
        // 词嵌入
        params += vocab_size * hidden_size;
        
        // Transformer层
        for (uint32_t i = 0; i < num_layers; i++) {
            // 自注意力: Q,K,V投影 + 输出投影
            params += 4 * hidden_size * hidden_size;
            
            // 前馈网络
            params += 2 * hidden_size * ffn_dim;
            
            // 层归一化（可忽略，参数量很少）
        }
        
        // 语言模型头
        params += hidden_size * vocab_size;
        
        return params;
    }
};

// 训练批次
struct TrainingBatch {
    Tensor input_ids;      // [batch_size, seq_len]
    Tensor attention_mask; // [batch_size, seq_len]
    Tensor labels;         // [batch_size, seq_len] 或 [batch_size]
    Tensor position_ids;   // [batch_size, seq_len]
    
    // 对于奖励建模
    Tensor chosen_ids;     // 被选中的回答
    Tensor rejected_ids;   // 被拒绝的回答
    scalar_t chosen_score; // 人工评分
    scalar_t rejected_score;
};

// 损失函数结果
struct LossResult {
    scalar_t loss_value;
    Tensor gradients;  // 可选，用于手动反向传播
    std::map<std::string, scalar_t> metrics;  // 额外指标
};

#endif // BASE_TYPES_HPP
```

## 2. 数据处理管道

```cpp
// data_pipeline.hpp
#ifndef DATA_PIPELINE_HPP
#define DATA_PIPELINE_HPP

#include "base_types.hpp"
#include <fstream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

// ==================== 数据处理流水线 ====================
// Data Processing Pipeline

class DataPipeline {
private:
    std::queue<TrainingBatch> data_queue;
    std::mutex queue_mutex;
    std::condition_variable data_ready;
    bool stop_flag = false;
    
    // 数据源
    std::vector<std::string> data_files;
    size_t current_file_idx = 0;
    
    // 分词器
    class Tokenizer* tokenizer;
    
    // 预处理配置
    struct {
        size_t max_seq_len = 1024;
        bool use_causal_mask = true;
        bool shuffle = true;
        size_t buffer_size = 10000;  // 预取缓冲区大小
    } config;
    
public:
    DataPipeline(const std::string& data_dir, 
                Tokenizer* tok, 
                size_t batch_size = 32) {
        tokenizer = tok;
        load_data_files(data_dir);
        
        // 启动数据加载线程
        std::thread loader(&DataPipeline::data_loader_thread, this);
        loader.detach();
    }
    
    // 获取下一个批次
    TrainingBatch get_batch() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // 等待数据就绪
        data_ready.wait(lock, [this]() { 
            return !data_queue.empty() || stop_flag; 
        });
        
        if (stop_flag && data_queue.empty()) {
            throw std::runtime_error("数据管道已停止");
        }
        
        TrainingBatch batch = data_queue.front();
        data_queue.pop();
        
        return batch;
    }
    
    // 数据加载线程
    void data_loader_thread() {
        std::vector<std::string> buffer;
        
        while (!stop_flag) {
            // 填充缓冲区
            while (buffer.size() < config.buffer_size && 
                   current_file_idx < data_files.size()) {
                std::string file_path = data_files[current_file_idx];
                load_file_to_buffer(file_path, buffer);
                current_file_idx = (current_file_idx + 1) % data_files.size();
            }
            
            // 打乱数据
            if (config.shuffle) {
                std::random_shuffle(buffer.begin(), buffer.end());
            }
            
            // 创建批次
            for (size_t i = 0; i + config.buffer_size <= buffer.size(); i += config.buffer_size) {
                std::vector<std::string> batch_texts(
                    buffer.begin() + i, 
                    buffer.begin() + i + config.buffer_size
                );
                
                TrainingBatch batch = create_training_batch(batch_texts);
                
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    data_queue.push(batch);
                }
                
                data_ready.notify_one();
            }
            
            // 清空已处理的数据
            buffer.clear();
        }
    }
    
private:
    void load_data_files(const std::string& data_dir) {
        // 递归扫描目录，收集所有文本文件
        // 这里简化为硬编码文件列表
        data_files = {
            data_dir + "/corpus_1.txt",
            data_dir + "/corpus_2.txt",
            // ...
        };
    }
    
    void load_file_to_buffer(const std::string& file_path, 
                           std::vector<std::string>& buffer) {
        std::ifstream file(file_path);
        std::string line;
        
        while (std::getline(file, line)) {
            if (!line.empty()) {
                buffer.push_back(line);
            }
        }
    }
    
    TrainingBatch create_training_batch(const std::vector<std::string>& texts) {
        TrainingBatch batch;
        
        size_t batch_size = texts.size();
        size_t seq_len = config.max_seq_len;
        
        // 初始化张量
        batch.input_ids = Tensor({batch_size, seq_len});
        batch.attention_mask = Tensor({batch_size, seq_len});
        batch.labels = Tensor({batch_size, seq_len});
        batch.position_ids = Tensor({batch_size, seq_len});
        
        // 填充数据
        for (size_t b = 0; b < batch_size; b++) {
            std::vector<token_id_t> tokens = tokenizer->encode(texts[b]);
            
            // 截断或填充到seq_len
            if (tokens.size() > seq_len) {
                tokens.resize(seq_len);
            } else if (tokens.size() < seq_len) {
                // 填充<pad> token（假设ID为0）
                tokens.resize(seq_len, 0);
            }
            
            for (size_t s = 0; s < seq_len; s++) {
                batch.input_ids[{b, s}] = tokens[s];
                batch.attention_mask[{b, s}] = (tokens[s] != 0) ? 1.0f : 0.0f;
                
                // 对于语言建模，标签是下一个token
                if (s < seq_len - 1) {
                    batch.labels[{b, s}] = tokens[s + 1];
                } else {
                    batch.labels[{b, s}] = -100;  // 忽略
                }
                
                batch.position_ids[{b, s}] = s;
            }
        }
        
        return batch;
    }
};

// ==================== 分词器实现 ====================
class Tokenizer {
private:
    std::map<std::string, token_id_t> token_to_id;
    std::map<token_id_t, std::string> id_to_token;
    token_id_t vocab_size = 0;
    
    // BPE合并规则
    std::map<std::pair<std::string, std::string>, token_id_t> merges;
    
public:
    Tokenizer(vocab_size_t size = 50257) : vocab_size(size) {
        initialize_base_vocab();
    }
    
    void initialize_base_vocab() {
        // 基础ASCII字符
        for (int i = 0; i < 256; i++) {
            std::string token(1, static_cast<char>(i));
            token_to_id[token] = i;
            id_to_token[i] = token;
        }
        
        // 特殊token
        token_to_id["<pad>"] = 256;
        token_to_id["<eos>"] = 257;
        token_to_id["<unk>"] = 258;
        
        id_to_token[256] = "<pad>";
        id_to_token[257] = "<eos>";
        id_to_token[258] = "<unk>";
    }
    
    std::vector<token_id_t> encode(const std::string& text) {
        // 简化的BPE编码
        std::vector<std::string> tokens;
        
        // 按字符分割
        for (char c : text) {
            tokens.push_back(std::string(1, c));
        }
        
        // 应用BPE合并
        bool changed = true;
        while (changed) {
            changed = false;
            
            // 找到最频繁的字符对
            std::map<std::pair<std::string, std::string>, int> pair_counts;
            for (size_t i = 0; i < tokens.size() - 1; i++) {
                auto pair = std::make_pair(tokens[i], tokens[i+1]);
                pair_counts[pair]++;
            }
            
            if (pair_counts.empty()) break;
            
            // 检查是否有合并规则
            for (const auto& [pair, count] : pair_counts) {
                if (merges.find(pair) != merges.end()) {
                    // 合并这对字符
                    std::vector<std::string> new_tokens;
                    for (size_t i = 0; i < tokens.size(); i++) {
                        if (i < tokens.size() - 1 && 
                            tokens[i] == pair.first && 
                            tokens[i+1] == pair.second) {
                            new_tokens.push_back(pair.first + pair.second);
                            i++;  // 跳过下一个
                        } else {
                            new_tokens.push_back(tokens[i]);
                        }
                    }
                    tokens = new_tokens;
                    changed = true;
                    break;
                }
            }
        }
        
        // 转换为ID
        std::vector<token_id_t> ids;
        for (const auto& token : tokens) {
            if (token_to_id.find(token) != token_to_id.end()) {
                ids.push_back(token_to_id[token]);
            } else {
                ids.push_back(token_to_id["<unk>"]);
            }
        }
        
        return ids;
    }
    
    std::string decode(const std::vector<token_id_t>& ids) {
        std::string text;
        for (token_id_t id : ids) {
            if (id_to_token.find(id) != id_to_token.end()) {
                text += id_to_token[id];
            } else {
                text += "<unk>";
            }
        }
        return text;
    }
};

#endif // DATA_PIPELINE_HPP
```

## 3. 预训练阶段（完整实现）

```cpp
// pretraining.hpp
#ifndef PRETRAINING_HPP
#define PRETRAINING_HPP

#include "base_types.hpp"
#include <cmath>
#include <vector>
#include <functional>

// ==================== 激活函数 ====================
namespace Activation {
    inline scalar_t relu(scalar_t x) {
        return std::max((scalar_t)0.0, x);
    }
    
    inline scalar_t gelu(scalar_t x) {
        // GELU近似实现
        return 0.5 * x * (1 + std::tanh(
            std::sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)
        ));
    }
    
    inline scalar_t softplus(scalar_t x) {
        return std::log(1 + std::exp(x));
    }
}

// ==================== 层归一化 ====================
class LayerNorm {
private:
    Tensor gamma;  // 缩放参数 [hidden_size]
    Tensor beta;   // 平移参数 [hidden_size]
    scalar_t eps;
    bool affine;   // 是否使用可学习参数
    
public:
    LayerNorm(hidden_dim_t hidden_size, scalar_t epsilon = 1e-5, bool aff = true) 
        : eps(epsilon), affine(aff) {
        gamma = Tensor({hidden_size}, true);
        beta = Tensor({hidden_size}, true);
        
        // 初始化
        for (size_t i = 0; i < hidden_size; i++) {
            gamma.data[i] = 1.0f;
            beta.data[i] = 0.0f;
        }
    }
    
    Tensor forward(const Tensor& x) {
        // x shape: [batch_size, seq_len, hidden_size]
        size_t batch_size = x.shape[0];
        size_t seq_len = x.shape[1];
        size_t hidden_size = x.shape[2];
        
        Tensor output(x.shape);
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                // 计算均值和方差
                scalar_t mean = 0.0f;
                scalar_t variance = 0.0f;
                
                for (size_t h = 0; h < hidden_size; h++) {
                    mean += x[{b, s, h}];
                }
                mean /= hidden_size;
                
                for (size_t h = 0; h < hidden_size; h++) {
                    scalar_t diff = x[{b, s, h}] - mean;
                    variance += diff * diff;
                }
                variance /= hidden_size;
                
                // 归一化
                scalar_t std = std::sqrt(variance + eps);
                
                for (size_t h = 0; h < hidden_size; h++) {
                    scalar_t normalized = (x[{b, s, h}] - mean) / std;
                    
                    if (affine) {
                        output[{b, s, h}] = gamma.data[h] * normalized + beta.data[h];
                    } else {
                        output[{b, s, h}] = normalized;
                    }
                }
            }
        }
        
        return output;
    }
    
    // 反向传播（简化版）
    void backward(const Tensor& grad_output, const Tensor& x) {
        // 计算gamma和beta的梯度
        // 实际实现需要完整的反向传播
    }
};

// ==================== 前馈网络 ====================
class FeedForward {
private:
    Tensor weight1;  // [hidden_size, ffn_dim]
    Tensor bias1;    // [ffn_dim]
    Tensor weight2;  // [ffn_dim, hidden_size]
    Tensor bias2;    // [hidden_size]
    
    hidden_dim_t hidden_size;
    hidden_dim_t ffn_dim;
    
public:
    FeedForward(hidden_dim_t h_size, hidden_dim_t f_dim) 
        : hidden_size(h_size), ffn_dim(f_dim) {
        
        // 初始化权重
        weight1 = Tensor({hidden_size, ffn_dim}, true);
        bias1 = Tensor({ffn_dim}, true);
        weight2 = Tensor({ffn_dim, hidden_size}, true);
        bias2 = Tensor({hidden_size}, true);
        
        initialize_weights();
    }
    
    void initialize_weights() {
        // Xavier/He初始化
        scalar_t std1 = std::sqrt(2.0f / (hidden_size + ffn_dim));
        scalar_t std2 = std::sqrt(2.0f / (ffn_dim + hidden_size));
        
        std::normal_distribution<scalar_t> dist1(0.0f, std1);
        std::normal_distribution<scalar_t> dist2(0.0f, std2);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (size_t i = 0; i < weight1.numel(); i++) {
            weight1.data[i] = dist1(gen);
        }
        for (size_t i = 0; i < bias1.numel(); i++) {
            bias1.data[i] = 0.0f;
        }
        for (size_t i = 0; i < weight2.numel(); i++) {
            weight2.data[i] = dist2(gen);
        }
        for (size_t i = 0; i < bias2.numel(); i++) {
            bias2.data[i] = 0.0f;
        }
    }
    
    Tensor forward(const Tensor& x) {
        // x shape: [batch_size, seq_len, hidden_size]
        size_t batch_size = x.shape[0];
        size_t seq_len = x.shape[1];
        
        // 第一层: x * W1 + b1
        Tensor hidden(Tensor({batch_size, seq_len, ffn_dim}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t f = 0; f < ffn_dim; f++) {
                    scalar_t sum = bias1.data[f];
                    
                    for (size_t h = 0; h < hidden_size; h++) {
                        sum += x[{b, s, h}] * weight1[{h, f}];
                    }
                    
                    hidden[{b, s, f}] = Activation::gelu(sum);
                }
            }
        }
        
        // 第二层: hidden * W2 + b2
        Tensor output(Tensor({batch_size, seq_len, hidden_size}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t h = 0; h < hidden_size; h++) {
                    scalar_t sum = bias2.data[h];
                    
                    for (size_t f = 0; f < ffn_dim; f++) {
                        sum += hidden[{b, s, f}] * weight2[{f, h}];
                    }
                    
                    output[{b, s, h}] = sum;
                }
            }
        }
        
        return output;
    }
};

// ==================== 多头注意力 ====================
class MultiHeadAttention {
private:
    Tensor W_q;  // [hidden_size, hidden_size]
    Tensor W_k;  // [hidden_size, hidden_size]
    Tensor W_v;  // [hidden_size, hidden_size]
    Tensor W_o;  // [hidden_size, hidden_size]
    
    hidden_dim_t hidden_size;
    uint32_t num_heads;
    hidden_dim_t head_dim;
    scalar_t dropout_rate;
    
public:
    MultiHeadAttention(hidden_dim_t h_size, uint32_t n_heads, scalar_t dropout = 0.1f)
        : hidden_size(h_size), num_heads(n_heads), dropout_rate(dropout) {
        
        head_dim = hidden_size / num_heads;
        
        // 初始化权重
        W_q = Tensor({hidden_size, hidden_size}, true);
        W_k = Tensor({hidden_size, hidden_size}, true);
        W_v = Tensor({hidden_size, hidden_size}, true);
        W_o = Tensor({hidden_size, hidden_size}, true);
        
        initialize_weights();
    }
    
    void initialize_weights() {
        scalar_t std = std::sqrt(2.0f / (hidden_size * 2));
        std::normal_distribution<scalar_t> dist(0.0f, std);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // 初始化所有权重
        auto init_tensor = [&](Tensor& t) {
            for (size_t i = 0; i < t.numel(); i++) {
                t.data[i] = dist(gen);
            }
        };
        
        init_tensor(W_q);
        init_tensor(W_k);
        init_tensor(W_v);
        init_tensor(W_o);
    }
    
    Tensor forward(const Tensor& x, const Tensor& attention_mask) {
        // x shape: [batch_size, seq_len, hidden_size]
        // mask shape: [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
        
        size_t batch_size = x.shape[0];
        size_t seq_len = x.shape[1];
        
        // 1. 线性投影得到Q,K,V
        Tensor Q = linear_projection(x, W_q);  // [batch, seq, hidden]
        Tensor K = linear_projection(x, W_k);
        Tensor V = linear_projection(x, W_v);
        
        // 2. 重塑为多头格式
        Tensor Q_heads = reshape_to_heads(Q);  // [batch, heads, seq, head_dim]
        Tensor K_heads = reshape_to_heads(K);
        Tensor V_heads = reshape_to_heads(V);
        
        // 3. 计算注意力分数
        Tensor attention_scores = compute_attention_scores(Q_heads, K_heads);
        
        // 4. 应用掩码
        if (attention_mask.shape.size() > 0) {
            apply_attention_mask(attention_scores, attention_mask);
        }
        
        // 5. Softmax得到注意力权重
        Tensor attention_weights = softmax_attention(attention_scores);
        
        // 6. 应用Dropout（训练时）
        // 这里省略了dropout实现
        
        // 7. 注意力加权
        Tensor attention_output = apply_attention(attention_weights, V_heads);
        
        // 8. 重塑回原始形状
        Tensor output = reshape_from_heads(attention_output);
        
        // 9. 输出投影
        output = linear_projection(output, W_o);
        
        return output;
    }
    
private:
    Tensor linear_projection(const Tensor& x, const Tensor& W) {
        size_t batch_size = x.shape[0];
        size_t seq_len = x.shape[1];
        
        Tensor result(Tensor({batch_size, seq_len, hidden_size}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t h = 0; h < hidden_size; h++) {
                    scalar_t sum = 0.0f;
                    
                    for (size_t i = 0; i < hidden_size; i++) {
                        sum += x[{b, s, i}] * W[{i, h}];
                    }
                    
                    result[{b, s, h}] = sum;
                }
            }
        }
        
        return result;
    }
    
    Tensor reshape_to_heads(const Tensor& x) {
        size_t batch_size = x.shape[0];
        size_t seq_len = x.shape[1];
        
        Tensor heads(Tensor({batch_size, num_heads, seq_len, head_dim}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t s = 0; s < seq_len; s++) {
                    for (size_t d = 0; d < head_dim; d++) {
                        size_t src_idx = h * head_dim + d;
                        heads[{b, h, s, d}] = x[{b, s, src_idx}];
                    }
                }
            }
        }
        
        return heads;
    }
    
    Tensor compute_attention_scores(const Tensor& Q, const Tensor& K) {
        size_t batch_size = Q.shape[0];
        size_t num_heads = Q.shape[1];
        size_t seq_len_q = Q.shape[2];
        size_t seq_len_k = K.shape[2];
        
        Tensor scores(Tensor({batch_size, num_heads, seq_len_q, seq_len_k}));
        
        scalar_t scale_factor = 1.0f / std::sqrt(static_cast<scalar_t>(head_dim));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < seq_len_q; i++) {
                    for (size_t j = 0; j < seq_len_k; j++) {
                        scalar_t score = 0.0f;
                        
                        for (size_t d = 0; d < head_dim; d++) {
                            score += Q[{b, h, i, d}] * K[{b, h, j, d}];
                        }
                        
                        scores[{b, h, i, j}] = score * scale_factor;
                    }
                }
            }
        }
        
        return scores;
    }
    
    void apply_attention_mask(Tensor& scores, const Tensor& mask) {
        size_t batch_size = scores.shape[0];
        size_t num_heads = scores.shape[1];
        size_t seq_len_q = scores.shape[2];
        size_t seq_len_k = scores.shape[3];
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < seq_len_q; i++) {
                    for (size_t j = 0; j < seq_len_k; j++) {
                        if (mask.shape.size() == 2) {
                            // 二维掩码
                            if (mask[{i, j}] == 0.0f) {
                                scores[{b, h, i, j}] = -1e9f;
                            }
                        } else if (mask.shape.size() == 3) {
                            // 三维掩码
                            if (mask[{b, i, j}] == 0.0f) {
                                scores[{b, h, i, j}] = -1e9f;
                            }
                        }
                    }
                }
            }
        }
    }
    
    Tensor softmax_attention(const Tensor& scores) {
        size_t batch_size = scores.shape[0];
        size_t num_heads = scores.shape[1];
        size_t seq_len_q = scores.shape[2];
        size_t seq_len_k = scores.shape[3];
        
        Tensor weights(Tensor({batch_size, num_heads, seq_len_q, seq_len_k}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < seq_len_q; i++) {
                    // 找到最大值（数值稳定性）
                    scalar_t max_val = -std::numeric_limits<scalar_t>::max();
                    for (size_t j = 0; j < seq_len_k; j++) {
                        max_val = std::max(max_val, scores[{b, h, i, j}]);
                    }
                    
                    // 计算指数和
                    scalar_t sum_exp = 0.0f;
                    for (size_t j = 0; j < seq_len_k; j++) {
                        sum_exp += std::exp(scores[{b, h, i, j}] - max_val);
                    }
                    
                    // 计算softmax
                    for (size_t j = 0; j < seq_len_k; j++) {
                        weights[{b, h, i, j}] = 
                            std::exp(scores[{b, h, i, j}] - max_val) / sum_exp;
                    }
                }
            }
        }
        
        return weights;
    }
    
    Tensor apply_attention(const Tensor& weights, const Tensor& V) {
        size_t batch_size = weights.shape[0];
        size_t num_heads = weights.shape[1];
        size_t seq_len_q = weights.shape[2];
        size_t seq_len_k = weights.shape[3];
        size_t head_dim = V.shape[3];
        
        Tensor output(Tensor({batch_size, num_heads, seq_len_q, head_dim}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < seq_len_q; i++) {
                    for (size_t d = 0; d < head_dim; d++) {
                        scalar_t sum = 0.0f;
                        
                        for (size_t j = 0; j < seq_len_k; j++) {
                            sum += weights[{b, h, i, j}] * V[{b, h, j, d}];
                        }
                        
                        output[{b, h, i, d}] = sum;
                    }
                }
            }
        }
        
        return output;
    }
    
    Tensor reshape_from_heads(const Tensor& x) {
        size_t batch_size = x.shape[0];
        size_t num_heads = x.shape[1];
        size_t seq_len = x.shape[2];
        size_t head_dim = x.shape[3];
        
        Tensor output(Tensor({batch_size, seq_len, hidden_size}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t h = 0; h < num_heads; h++) {
                    for (size_t d = 0; d < head_dim; d++) {
                        size_t dst_idx = h * head_dim + d;
                        output[{b, s, dst_idx}] = x[{b, h, s, d}];
                    }
                }
            }
        }
        
        return output;
    }
};

// ==================== Transformer块 ====================
class TransformerBlock {
private:
    MultiHeadAttention attention;
    LayerNorm norm1;
    FeedForward ff;
    LayerNorm norm2;
    scalar_t dropout_rate;
    
public:
    TransformerBlock(const ModelConfig& config)
        : attention(config.hidden_size, config.num_heads, config.dropout_rate),
          norm1(config.hidden_size),
          ff(config.hidden_size, config.ffn_dim),
          norm2(config.hidden_size),
          dropout_rate(config.dropout_rate) {}
    
    Tensor forward(const Tensor& x, const Tensor& attention_mask) {
        // 自注意力 + 残差连接 + 层归一化
        Tensor attn_output = attention.forward(x, attention_mask);
        Tensor x1 = add_residual(x, attn_output);
        x1 = norm1.forward(x1);
        
        // 前馈网络 + 残差连接 + 层归一化
        Tensor ff_output = ff.forward(x1);
        Tensor output = add_residual(x1, ff_output);
        output = norm2.forward(output);
        
        return output;
    }
    
private:
    Tensor add_residual(const Tensor& x, const Tensor& residual) {
        // 简单的逐元素相加
        Tensor result(x.shape);
        
        for (size_t i = 0; i < x.numel(); i++) {
            result.data[i] = x.data[i] + residual.data[i];
        }
        
        return result;
    }
};

// ==================== GPT模型 ====================
class GPTModel {
private:
    ModelConfig config;
    
    // 嵌入层
    Tensor token_embedding;  // [vocab_size, hidden_size]
    Tensor position_embedding;  // [max_seq_len, hidden_size]
    
    // Transformer层
    std::vector<TransformerBlock> layers;
    
    // 输出层
    LayerNorm final_norm;
    Tensor lm_head;  // [hidden_size, vocab_size]
    
public:
    GPTModel(const ModelConfig& cfg) : config(cfg), final_norm(cfg.hidden_size) {
        // 初始化嵌入层
        token_embedding = Tensor({config.vocab_size, config.hidden_size}, true);
        position_embedding = Tensor({config.max_seq_len, config.hidden_size}, true);
        
        // 初始化Transformer层
        for (uint32_t i = 0; i < config.num_layers; i++) {
            layers.emplace_back(config);
        }
        
        // 初始化语言模型头
        lm_head = Tensor({config.hidden_size, config.vocab_size}, true);
        
        initialize_weights();
    }
    
    void initialize_weights() {
        // 初始化所有参数
        scalar_t std = 0.02f;  // GPT-2使用的标准差
        
        std::normal_distribution<scalar_t> dist(0.0f, std);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // 初始化词嵌入
        for (size_t i = 0; i < token_embedding.numel(); i++) {
            token_embedding.data[i] = dist(gen);
        }
        
        // 初始化位置嵌入（使用正弦余弦）
        for (size_t pos = 0; pos < config.max_seq_len; pos++) {
            for (size_t i = 0; i < config.hidden_size; i++) {
                if (i % 2 == 0) {
                    position_embedding[{pos, i}] = 
                        std::sin(pos / std::pow(10000.0f, i / config.hidden_size));
                } else {
                    position_embedding[{pos, i}] = 
                        std::cos(pos / std::pow(10000.0f, (i-1) / config.hidden_size));
                }
            }
        }
        
        // 初始化lm_head（与词嵌入共享权重，这是常见做法）
        // 这里简化为随机初始化
        for (size_t i = 0; i < lm_head.numel(); i++) {
            lm_head.data[i] = dist(gen);
        }
    }
    
    Tensor forward(const Tensor& input_ids, const Tensor& attention_mask) {
        size_t batch_size = input_ids.shape[0];
        size_t seq_len = input_ids.shape[1];
        
        // 1. 词嵌入 + 位置嵌入
        Tensor embeddings(Tensor({batch_size, seq_len, config.hidden_size}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                token_id_t token_id = static_cast<token_id_t>(input_ids[{b, s}]);
                
                for (size_t h = 0; h < config.hidden_size; h++) {
                    embeddings[{b, s, h}] = 
                        token_embedding[{token_id, h}] + 
                        position_embedding[{s, h}];
                }
            }
        }
        
        // 2. 通过Transformer层
        Tensor hidden_states = embeddings;
        
        for (auto& layer : layers) {
            hidden_states = layer.forward(hidden_states, attention_mask);
        }
        
        // 3. 最终层归一化
        hidden_states = final_norm.forward(hidden_states);
        
        // 4. 语言模型头
        Tensor logits(Tensor({batch_size, seq_len, config.vocab_size}));
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t v = 0; v < config.vocab_size; v++) {
                    scalar_t sum = 0.0f;
                    
                    for (size_t h = 0; h < config.hidden_size; h++) {
                        sum += hidden_states[{b, s, h}] * lm_head[{h, v}];
                    }
                    
                    logits[{b, s, v}] = sum;
                }
            }
        }
        
        return logits;
    }
};

#endif // PRETRAINING_HPP
```

## 4. 优化器和损失函数

```cpp
// optimizer.hpp
#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "base_types.hpp"
#include <vector>
#include <cmath>
#include <map>

// ==================== 损失函数 ====================
class LossFunction {
public:
    virtual LossResult compute(const Tensor& predictions, 
                              const Tensor& targets) = 0;
    virtual ~LossFunction() = default;
};

class CrossEntropyLoss : public LossFunction {
private:
    scalar_t label_smoothing;
    
public:
    CrossEntropyLoss(scalar_t smoothing = 0.0f) : label_smoothing(smoothing) {}
    
    LossResult compute(const Tensor& predictions, const Tensor& targets) override {
        // predictions: [batch_size, seq_len, vocab_size]
        // targets: [batch_size, seq_len]
        
        size_t batch_size = predictions.shape[0];
        size_t seq_len = predictions.shape[1];
        size_t vocab_size = predictions.shape[2];
        
        LossResult result;
        scalar_t total_loss = 0.0f;
        size_t total_tokens = 0;
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                token_id_t target_id = static_cast<token_id_t>(targets[{b, s}]);
                
                if (target_id == static_cast<token_id_t>(-100)) {
                    continue;  // 忽略标记
                }
                
                // 找到预测的最大值（数值稳定性）
                scalar_t max_logit = -std::numeric_limits<scalar_t>::max();
                for (size_t v = 0; v < vocab_size; v++) {
                    max_logit = std::max(max_logit, predictions[{b, s, v}]);
                }
                
                // 计算log sum exp
                scalar_t log_sum_exp = 0.0f;
                for (size_t v = 0; v < vocab_size; v++) {
                    log_sum_exp += std::exp(predictions[{b, s, v}] - max_logit);
                }
                log_sum_exp = max_logit + std::log(log_sum_exp);
                
                // 计算交叉熵损失
                scalar_t target_logit = predictions[{b, s, target_id}];
                scalar_t loss = log_sum_exp - target_logit;
                
                // 标签平滑
                if (label_smoothing > 0.0f) {
                    scalar_t smooth_loss = 0.0f;
                    for (size_t v = 0; v < vocab_size; v++) {
                        if (v == target_id) {
                            smooth_loss += (1.0f - label_smoothing) * 
                                         (log_sum_exp - predictions[{b, s, v}]);
                        } else {
                            smooth_loss += (label_smoothing / (vocab_size - 1)) * 
                                         (log_sum_exp - predictions[{b, s, v}]);
                        }
                    }
                    loss = smooth_loss;
                }
                
                total_loss += loss;
                total_tokens++;
            }
        }
        
        result.loss_value = total_loss / total_tokens;
        result.metrics["perplexity"] = std::exp(result.loss_value);
        
        return result;
    }
};

// ==================== 优化器 ====================
class Optimizer {
protected:
    scalar_t learning_rate;
    std::vector<Tensor*> parameters;
    std::vector<OptimizerState> states;
    
public:
    Optimizer(scalar_t lr = 1e-3) : learning_rate(lr) {}
    
    virtual void add_parameters(Tensor* param) {
        parameters.push_back(param);
        states.emplace_back(learning_rate);
    }
    
    virtual void step() = 0;
    virtual void zero_grad() {
        for (auto param : parameters) {
            if (param->requires_grad) {
                std::fill(param->grad.begin(), param->grad.end(), 0.0f);
            }
        }
    }
    
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
private:
    scalar_t momentum;
    std::vector<std::vector<scalar_t>> velocities;
    
public:
    SGD(scalar_t lr = 1e-3, scalar_t mom = 0.9f) 
        : Optimizer(lr), momentum(mom) {}
    
    void add_parameters(Tensor* param) override {
        Optimizer::add_parameters(param);
        velocities.emplace_back(param->grad.size(), 0.0f);
    }
    
    void step() override {
        for (size_t i = 0; i < parameters.size(); i++) {
            Tensor* param = parameters[i];
            
            if (!param->requires_grad) continue;
            
            for (size_t j = 0; j < param->grad.size(); j++) {
                // 动量更新
                velocities[i][j] = momentum * velocities[i][j] + 
                                  learning_rate * param->grad[j];
                param->data[j] -= velocities[i][j];
            }
        }
    }
};

class AdamW : public Optimizer {
private:
    scalar_t beta1;
    scalar_t beta2;
    scalar_t epsilon;
    scalar_t weight_decay;
    
public:
    AdamW(scalar_t lr = 1e-3, scalar_t b1 = 0.9f, scalar_t b2 = 0.999f,
         scalar_t eps = 1e-8f, scalar_t wd = 0.01f)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd) {}
    
    void step() override {
        for (size_t i = 0; i < parameters.size(); i++) {
            Tensor* param = parameters[i];
            OptimizerState& state = states[i];
            
            if (!param->requires_grad) continue;
            
            state.step++;
            
            // 初始化矩估计
            if (state.m.empty()) {
                state.m.resize(param->grad.size(), 0.0f);
                state.v.resize(param->grad.size(), 0.0f);
            }
            
            // 偏差校正系数
            scalar_t m_correction = 1.0f - std::pow(beta1, state.step);
            scalar_t v_correction = 1.0f - std::pow(beta2, state.step);
            
            for (size_t j = 0; j < param->grad.size(); j++) {
                scalar_t grad = param->grad[j];
                
                // 更新一阶矩
                state.m[j] = beta1 * state.m[j] + (1.0f - beta1) * grad;
                
                // 更新二阶矩
                state.v[j] = beta2 * state.v[j] + (1.0f - beta2) * grad * grad;
                
                // 偏差校正
                scalar_t m_hat = state.m[j] / m_correction;
                scalar_t v_hat = state.v[j] / v_correction;
                
                // AdamW更新规则
                scalar_t update = state.learning_rate * m_hat / 
                                 (std::sqrt(v_hat) + epsilon);
                
                // 权重衰减
                update += state.learning_rate * weight_decay * param->data[j];
                
                param->data[j] -= update;
            }
        }
    }
};

// ==================== 梯度裁剪 ====================
class GradientClipper {
private:
    scalar_t max_norm;
    
public:
    GradientClipper(scalar_t max_n = 1.0f) : max_norm(max_n) {}
    
    void clip(std::vector<Tensor*>& parameters) {
        // 计算总梯度范数
        scalar_t total_norm_sq = 0.0f;
        
        for (auto param : parameters) {
            if (!param->requires_grad) continue;
            
            for (scalar_t grad : param->grad) {
                total_norm_sq += grad * grad;
            }
        }
        
        scalar_t total_norm = std::sqrt(total_norm_sq);
        
        // 如果超过最大范数，进行缩放
        if (total_norm > max_norm) {
            scalar_t scale = max_norm / (total_norm + 1e-6f);
            
            for (auto param : parameters) {
                if (!param->requires_grad) continue;
                
                for (scalar_t& grad : param->grad) {
                    grad *= scale;
                }
            }
        }
    }
};

// ==================== 学习率调度器 ====================
class LearningRateScheduler {
public:
    enum ScheduleType {
        CONSTANT,
        LINEAR_WARMUP,
        COSINE_DECAY,
        STEP_DECAY
    };
    
private:
    ScheduleType type;
    scalar_t initial_lr;
    scalar_t current_lr;
    scalar_t min_lr;
    size_t warmup_steps;
    size_t total_steps;
    size_t current_step;
    
public:
    LearningRateScheduler(ScheduleType t = LINEAR_WARMUP, 
                         scalar_t lr = 1e-3,
                         size_t warmup = 1000,
                         size_t total = 100000,
                         scalar_t min = 1e-6f)
        : type(t), initial_lr(lr), current_lr(lr), min_lr(min),
          warmup_steps(warmup), total_steps(total), current_step(0) {}
    
    scalar_t get_lr() {
        current_step++;
        
        switch (type) {
            case CONSTANT:
                return initial_lr;
                
            case LINEAR_WARMUP: {
                if (current_step <= warmup_steps) {
                    // 线性预热
                    return initial_lr * (current_step / (scalar_t)warmup_steps);
                } else {
                    // 余弦衰减
                    scalar_t progress = (current_step - warmup_steps) / 
                                      (scalar_t)(total_steps - warmup_steps);
                    progress = std::min(progress, 1.0f);
                    
                    scalar_t cosine_decay = 0.5f * 
                        (1.0f + std::cos(M_PI * progress));
                    
                    current_lr = min_lr + 
                        (initial_lr - min_lr) * cosine_decay;
                    return current_lr;
                }
            }
                
            case COSINE_DECAY: {
                scalar_t progress = current_step / (scalar_t)total_steps;
                progress = std::min(progress, 1.0f);
                
                scalar_t cosine_decay = 0.5f * 
                    (1.0f + std::cos(M_PI * progress));
                
                current_lr = min_lr + 
                    (initial_lr - min_lr) * cosine_decay;
                return current_lr;
            }
                
            case STEP_DECAY: {
                // 每10000步衰减为原来的0.9倍
                size_t decay_steps = 10000;
                size_t decay_count = current_step / decay_steps;
                
                current_lr = initial_lr * std::pow(0.9f, decay_count);
                current_lr = std::max(current_lr, min_lr);
                return current_lr;
            }
        }
        
        return initial_lr;
    }
    
    void update_optimizer(Optimizer& optimizer) {
        scalar_t lr = get_lr();
        // 更新优化器的学习率
        // 这里需要根据具体的优化器实现来设置
    }
};

#endif // OPTIMIZER_HPP
```

## 5. 预训练主循环

```cpp
// train_pipeline.hpp
#ifndef TRAIN_PIPELINE_HPP
#define TRAIN_PIPELINE_HPP

#include "pretraining.hpp"
#include "optimizer.hpp"
#include "data_pipeline.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>

// ==================== 训练监控器 ====================
class TrainingMonitor {
private:
    struct Checkpoint {
        size_t step;
        scalar_t loss;
        scalar_t learning_rate;
        std::chrono::system_clock::time_point timestamp;
        std::string file_path;
    };
    
    std::vector<Checkpoint> checkpoints;
    std::string log_file;
    std::ofstream log_stream;
    
    // 训练统计
    scalar_t running_loss = 0.0f;
    size_t running_steps = 0;
    std::chrono::steady_clock::time_point start_time;
    
public:
    TrainingMonitor(const std::string& log_path = "training_log.txt") 
        : log_file(log_path) {
        log_stream.open(log_file);
        start_time = std::chrono::steady_clock::now();
    }
    
    ~TrainingMonitor() {
        if (log_stream.is_open()) {
            log_stream.close();
        }
    }
    
    void log_step(size_t step, scalar_t loss, scalar_t lr, 
                 const std::map<std::string, scalar_t>& metrics = {}) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - start_time).count();
        
        // 更新运行统计
        running_loss += loss;
        running_steps++;
        
        // 每100步打印一次
        if (step % 100 == 0) {
            scalar_t avg_loss = running_loss / running_steps;
            
            std::cout << std::setw(8) << step << " | "
                      << std::setw(10) << std::fixed << std::setprecision(4) 
                      << loss << " | "
                      << std::setw(10) << avg_loss << " | "
                      << std::setw(12) << std::scientific << lr << " | "
                      << std::setw(8) << elapsed << "s"
                      << std::endl;
            
            // 重置运行统计
            running_loss = 0.0f;
            running_steps = 0;
        }
        
        // 写入日志文件
        log_stream << step << ", " << loss << ", " << lr << ", " << elapsed;
        for (const auto& [key, value] : metrics) {
            log_stream << ", " << value;
        }
        log_stream << std::endl;
    }
    
    void save_checkpoint(size_t step, scalar_t loss, scalar_t lr, 
                        const std::string& model_path) {
        Checkpoint checkpoint;
        checkpoint.step = step;
        checkpoint.loss = loss;
        checkpoint.learning_rate = lr;
        checkpoint.timestamp = std::chrono::system_clock::now();
        checkpoint.file_path = model_path;
        
        checkpoints.push_back(checkpoint);
        
        // 保存检查点到文件
        std::ofstream checkpoint_file("checkpoints/checkpoint_" + 
                                     std::to_string(step) + ".json");
        
        checkpoint_file << "{"
                       << "\"step\": " << step << ", "
                       << "\"loss\": " << loss << ", "
                       << "\"lr\": " << lr << ", "
                       << "\"timestamp\": \"" 
                       << std::chrono::system_clock::to_time_t(checkpoint.timestamp)
                       << "\", "
                       << "\"file_path\": \"" << model_path << "\""
                       << "}" << std::endl;
        
        checkpoint_file.close();
    }
};

// ==================== 预训练主循环 ====================
class PreTrainingPipeline {
private:
    ModelConfig config;
    GPTModel model;
    DataPipeline data_pipeline;
    AdamW optimizer;
    CrossEntropyLoss loss_function;
    GradientClipper gradient_clipper;
    LearningRateScheduler lr_scheduler;
    TrainingMonitor monitor;
    
    // 训练状态
    size_t current_step = 0;
    size_t total_steps;
    size_t save_every;
    size_t eval_every;
    
public:
    PreTrainingPipeline(const ModelConfig& cfg, 
                       const std::string& data_dir,
                       size_t total_s = 100000,
                       size_t save_interval = 1000,
                       size_t eval_interval = 500)
        : config(cfg),
          model(cfg),
          data_pipeline(data_dir, new Tokenizer(cfg.vocab_size)),
          optimizer(1e-4, 0.9, 0.999, 1e-8, 0.01),
          loss_function(0.1f),  // 10%标签平滑
          gradient_clipper(1.0f),
          lr_scheduler(LearningRateScheduler::LINEAR_WARMUP, 
                      1e-4, 1000, total_s, 1e-6),
          monitor("pretraining_log.txt"),
          total_steps(total_s),
          save_every(save_interval),
          eval_every(eval_interval) {
        
        // 注册模型参数到优化器
        register_model_parameters();
    }
    
    void register_model_parameters() {
        // 这里需要注册模型的所有可训练参数
        // 由于我们简化了模型实现，这里省略具体实现
        // 在实际实现中，需要遍历模型的所有层，收集所有requires_grad=true的张量
    }
    
    void train() {
        std::cout << "开始预训练..." << std::endl;
        std::cout << "模型参数量: " << config.total_params() << std::endl;
        std::cout << "总训练步数: " << total_steps << std::endl;
        std::cout << "=" << 80 << std::endl;
        std::cout << std::setw(8) << "Step" << " | "
                  << std::setw(10) << "Loss" << " | "
                  << std::setw(10) << "Avg Loss" << " | "
                  << std::setw(12) << "LR" << " | "
                  << std::setw(8) << "Time" << std::endl;
        std::cout << "=" << 80 << std::endl;
        
        while (current_step < total_steps) {
            train_step();
            current_step++;
            
            // 定期评估
            if (current_step % eval_every == 0) {
                evaluate();
            }
            
            // 定期保存检查点
            if (current_step % save_every == 0) {
                save_checkpoint();
            }
        }
        
        std::cout << "预训练完成!" << std::endl;
    }
    
private:
    void train_step() {
        // 1. 获取训练批次
        TrainingBatch batch = data_pipeline.get_batch();
        
        // 2. 前向传播
        Tensor logits = model.forward(batch.input_ids, batch.attention_mask);
        
        // 3. 计算损失
        LossResult loss_result = loss_function.compute(logits, batch.labels);
        scalar_t loss = loss_result.loss_value;
        
        // 4. 反向传播（简化版）
        // 在实际实现中，这里需要计算梯度
        compute_gradients(logits, batch.labels);
        
        // 5. 梯度裁剪
        // gradient_clipper.clip(model_parameters);
        
        // 6. 更新学习率
        scalar_t lr = lr_scheduler.get_lr();
        // optimizer.set_learning_rate(lr);
        
        // 7. 优化器步骤
        optimizer.zero_grad();
        optimizer.step();
        
        // 8. 记录日志
        std::map<std::string, scalar_t> metrics;
        metrics["perplexity"] = loss_result.metrics["perplexity"];
        
        monitor.log_step(current_step, loss, lr, metrics);
    }
    
    void compute_gradients(const Tensor& logits, const Tensor& targets) {
        // 简化的梯度计算（仅用于演示）
        // 实际实现需要完整的反向传播
        // 这里我们假设已经计算好了梯度
    }
    
    void evaluate() {
        // 评估模型性能
        // 在实际实现中，这里应该在验证集上计算损失和指标
        
        std::cout << "评估步数 " << current_step << "..." << std::endl;
        
        // 示例：计算验证损失
        scalar_t valid_loss = 0.0f;
        size_t valid_steps = 10;  // 只评估少量批次
        
        for (size_t i = 0; i < valid_steps; i++) {
            TrainingBatch batch = data_pipeline.get_batch();
            Tensor logits = model.forward(batch.input_ids, batch.attention_mask);
            LossResult loss_result = loss_function.compute(logits, batch.labels);
            valid_loss += loss_result.loss_value;
        }
        
        valid_loss /= valid_steps;
        
        std::cout << "验证损失: " << valid_loss 
                  << ", 困惑度: " << std::exp(valid_loss) << std::endl;
    }
    
    void save_checkpoint() {
        std::string checkpoint_path = "checkpoints/step_" + 
                                     std::to_string(current_step) + ".bin";
        
        // 保存模型状态
        save_model(checkpoint_path);
        
        // 记录检查点
        monitor.save_checkpoint(current_step, 0.0f,  // 损失需要从监控器获取
                               lr_scheduler.get_lr(), checkpoint_path);
        
        std::cout << "保存检查点到: " << checkpoint_path << std::endl;
    }
    
    void save_model(const std::string& path) {
        // 保存模型参数到文件
        // 在实际实现中，这里需要序列化所有参数
        
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "无法打开文件保存模型: " << path << std::endl;
            return;
        }
        
        // 保存模型配置
        file.write(reinterpret_cast<const char*>(&config.vocab_size), 
                  sizeof(config.vocab_size));
        file.write(reinterpret_cast<const char*>(&config.hidden_size), 
                  sizeof(config.hidden_size));
        // ... 保存其他配置
        
        // 保存模型参数
        // 这里需要保存所有层的权重和偏置
        
        file.close();
    }
};

#endif // TRAIN_PIPELINE_HPP
```

## 6. SFT训练阶段

```cpp
// sft_training.hpp
#ifndef SFT_TRAINING_HPP
#define SFT_TRAINING_HPP

#include "pretraining.hpp"
#include "optimizer.hpp"

// ==================== SFT数据集 ====================
class SFTDataset {
private:
    struct InstructionExample {
        std::string instruction;
        std::string input;
        std::string output;
        std::string system_prompt;
    };
    
    std::vector<InstructionExample> examples;
    Tokenizer* tokenizer;
    
public:
    SFTDataset(Tokenizer* tok) : tokenizer(tok) {
        // 加载或生成SFT数据
        load_examples();
    }
    
    void load_examples() {
        // 示例数据
        examples = {
            {"解释什么是机器学习", "", 
             "机器学习是人工智能的一个分支，它使计算机能够从数据中学习...",
             "你是一个有帮助的AI助手"},
             
            {"写一首关于春天的诗", "",
             "春风拂面花香溢，万物复苏生机勃...",
             "你是一个富有诗意的AI"},
             
            {"如何煮意大利面？", "",
             "首先，把水烧开，加入盐...",
             "你是一个厨艺助手"},
             
            {"解释量子计算", "",
             "量子计算是利用量子力学原理进行计算的一种方法...",
             "你是一个科学助手"}
        };
    }
    
    TrainingBatch get_batch(size_t batch_size) {
        TrainingBatch batch;
        
        // 随机选择样本
        std::vector<size_t> indices(batch_size);
        for (size_t i = 0; i < batch_size; i++) {
            indices[i] = rand() % examples.size();
        }
        
        // 构建批次
        size_t max_seq_len = 512;
        
        batch.input_ids = Tensor({batch_size, max_seq_len});
        batch.attention_mask = Tensor({batch_size, max_seq_len});
        batch.labels = Tensor({batch_size, max_seq_len});
        
        for (size_t i = 0; i < batch_size; i++) {
            const InstructionExample& example = examples[indices[i]];
            
            // 格式化提示
            std::string prompt = format_prompt(example);
            
            // 编码提示和回复
            std::vector<token_id_t> prompt_tokens = tokenizer->encode(prompt);
            std::vector<token_id_t> output_tokens = tokenizer->encode(example.output);
            
            // 合并输入和输出
            std::vector<token_id_t> all_tokens = prompt_tokens;
            all_tokens.insert(all_tokens.end(), 
                            output_tokens.begin(), 
                            output_tokens.end());
            
            // 添加EOS token
            all_tokens.push_back(tokenizer->get_eos_id());
            
            // 截断或填充
            if (all_tokens.size() > max_seq_len) {
                all_tokens.resize(max_seq_len);
            } else if (all_tokens.size() < max_seq_len) {
                all_tokens.resize(max_seq_len, tokenizer->get_pad_id());
            }
            
            // 填充到批次
            for (size_t j = 0; j < max_seq_len; j++) {
                batch.input_ids[{i, j}] = all_tokens[j];
                batch.attention_mask[{i, j}] = (all_tokens[j] != 
                                              tokenizer->get_pad_id()) ? 1.0f : 0.0f;
                
                // 对于SFT，我们只计算输出部分的损失
                // 标签中，输入部分设置为-100（忽略）
                if (j < prompt_tokens.size()) {
                    batch.labels[{i, j}] = -100;
                } else if (j < all_tokens.size() - 1) {
                    // 输出部分的标签是下一个token
                    batch.labels[{i, j}] = all_tokens[j + 1];
                } else {
                    batch.labels[{i, j}] = -100;
                }
            }
        }
        
        return batch;
    }
    
private:
    std::string format_prompt(const InstructionExample& example) {
        // 使用ChatML格式
        std::string prompt;
        
        if (!example.system_prompt.empty()) {
            prompt += "<|im_start|>system\n";
            prompt += example.system_prompt + "\n";
            prompt += "<|im_end|>\n";
        }
        
        prompt += "<|im_start|>user\n";
        prompt += example.instruction;
        if (!example.input.empty()) {
            prompt += "\n" + example.input;
        }
        prompt += "\n<|im_end|>\n";
        prompt += "<|im_start|>assistant\n";
        
        return prompt;
    }
};

// ==================== SFT训练器 ====================
class SFTTrainer {
private:
    GPTModel& model;
    SFTDataset dataset;
    AdamW optimizer;
    CrossEntropyLoss loss_function;
    size_t num_epochs;
    
public:
    SFTTrainer(GPTModel& m, Tokenizer* tokenizer, 
               size_t epochs = 3, scalar_t lr = 2e-5)
        : model(m), dataset(tokenizer), 
          optimizer(lr, 0.9, 0.999, 1e-8, 0.01),
          num_epochs(epochs) {}
    
    void train() {
        std::cout << "开始SFT训练..." << std::endl;
        std::cout << "训练轮数: " << num_epochs << std::endl;
        
        size_t batch_size = 8;
        size_t total_steps = (dataset.size() / batch_size) * num_epochs;
        size_t current_step = 0;
        
        for (size_t epoch = 0; epoch < num_epochs; epoch++) {
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
            
            // 打乱数据
            dataset.shuffle();
            
            size_t num_batches = dataset.size() / batch_size;
            
            for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
                TrainingBatch batch = dataset.get_batch(batch_size);
                
                // 前向传播
                Tensor logits = model.forward(batch.input_ids, batch.attention_mask);
                
                // 计算损失（只计算assistant部分的损失）
                LossResult loss_result = loss_function.compute(logits, batch.labels);
                
                // 反向传播和优化
                optimizer.zero_grad();
                // compute_gradients(...);  // 计算梯度
                optimizer.step();
                
                current_step++;
                
                // 打印进度
                if (current_step % 100 == 0) {
                    std::cout << "Step " << current_step << "/" << total_steps
                              << ", Loss: " << loss_result.loss_value
                              << ", Perplexity: " << loss_result.metrics["perplexity"]
                              << std::endl;
                }
            }
        }
        
        std::cout << "SFT训练完成!" << std::endl;
    }
};

#endif // SFT_TRAINING_HPP
```

## 7. 奖励建模和RLHF

```cpp
// rlhf_training.hpp
#ifndef RLHF_TRAINING_HPP
#define RLHF_TRAINING_HPP

#include "sft_training.hpp"

// ==================== 奖励模型 ====================
class RewardModel {
private:
    GPTModel& base_model;  // SFT后的模型
    Tensor reward_head;    // [hidden_size, 1]
    
public:
    RewardModel(GPTModel& model) : base_model(model) {
        // 初始化奖励头
        reward_head = Tensor({model.config.hidden_size, 1}, true);
        
        // 随机初始化
        std::normal_distribution<scalar_t> dist(0.0f, 0.02f);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (size_t i = 0; i < reward_head.numel(); i++) {
            reward_head.data[i] = dist(gen);
        }
    }
    
    scalar_t forward(const Tensor& input_ids, const Tensor& attention_mask) {
        // 获取最后一个token的隐藏状态
        Tensor logits = base_model.forward(input_ids, attention_mask);
        
        // 假设logits的shape是[batch, seq, hidden]
        size_t batch_size = logits.shape[0];
        size_t seq_len = logits.shape[1];
        
        // 取最后一个token的隐藏状态
        Tensor last_hidden({batch_size, logits.shape[2]});
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < logits.shape[2]; h++) {
                last_hidden[{b, h}] = logits[{b, seq_len - 1, h}];
            }
        }
        
        // 通过奖励头
        scalar_t reward = 0.0f;
        for (size_t h = 0; h < logits.shape[2]; h++) {
            reward += last_hidden[{0, h}] * reward_head[{h, 0}];
        }
        
        return reward;
    }
};

// ==================== 奖励模型训练 ====================
class RewardModelTrainer {
private:
    RewardModel& reward_model;
    AdamW optimizer;
    
    // 偏好数据集
    struct PreferenceExample {
        std::string prompt;
        std::string chosen_response;    // 被选中的回答
        std::string rejected_response;  // 被拒绝的回答
        scalar_t chosen_score;          // 人工评分
        scalar_t rejected_score;
    };
    
    std::vector<PreferenceExample> dataset;
    Tokenizer* tokenizer;
    
public:
    RewardModelTrainer(RewardModel& rm, Tokenizer* tok)
        : reward_model(rm), optimizer(1e-5, 0.9, 0.999, 1e-8, 0.01), 
          tokenizer(tok) {
        load_preference_data();
    }
    
    void load_preference_data() {
        // 示例偏好数据
        dataset = {
            {"解释量子计算", 
             "量子计算是利用量子力学原理进行计算的方法...",
             "量子计算就是很快的计算",
             0.9f, 0.2f},
             
            {"写一首关于春天的诗",
             "春风拂面花香溢，万物复苏生机勃...",
             "春天来了，花开了",
             0.8f, 0.3f}
        };
    }
    
    scalar_t compute_preference_loss(scalar_t chosen_reward, 
                                    scalar_t rejected_reward) {
        // 成对排名损失
        // 目标：chosen_reward > rejected_reward
        
        // Bradley-Terry模型损失
        scalar_t loss = -std::log(1.0f / (1.0f + 
                          std::exp(rejected_reward - chosen_reward)));
        
        return loss;
    }
    
    void train_step() {
        // 随机选择一个偏好样本
        size_t idx = rand() % dataset.size();
        const PreferenceExample& example = dataset[idx];
        
        // 编码chosen响应
        std::string chosen_input = example.prompt + "\n" + example.chosen_response;
        std::vector<token_id_t> chosen_tokens = tokenizer->encode(chosen_input);
        
        // 编码rejected响应
        std::string rejected_input = example.prompt + "\n" + example.rejected_response;
        std::vector<token_id_t> rejected_tokens = tokenizer->encode(rejected_input);
        
        // 创建张量（简化）
        Tensor chosen_ids({1, (size_t)chosen_tokens.size()});
        Tensor chosen_mask({1, (size_t)chosen_tokens.size()});
        
        Tensor rejected_ids({1, (size_t)rejected_tokens.size()});
        Tensor rejected_mask({1, (size_t)rejected_tokens.size()});
        
        // 填充数据
        for (size_t i = 0; i < chosen_tokens.size(); i++) {
            chosen_ids[{0, i}] = chosen_tokens[i];
            chosen_mask[{0, i}] = 1.0f;
        }
        
        for (size_t i = 0; i < rejected_tokens.size(); i++) {
            rejected_ids[{0, i}] = rejected_tokens[i];
            rejected_mask[{0, i}] = 1.0f;
        }
        
        // 前向传播
        scalar_t chosen_reward = reward_model.forward(chosen_ids, chosen_mask);
        scalar_t rejected_reward = reward_model.forward(rejected_ids, rejected_mask);
        
        // 计算损失
        scalar_t loss = compute_preference_loss(chosen_reward, rejected_reward);
        
        // 反向传播和优化
        optimizer.zero_grad();
        // compute_gradients(...);
        optimizer.step();
        
        std::cout << "Reward Model Loss: " << loss 
                  << ", Chosen: " << chosen_reward 
                  << ", Rejected: " << rejected_reward << std::endl;
    }
};

// ==================== PPO算法 ====================
class PPO {
private:
    GPTModel& policy_model;   // 要优化的模型
    GPTModel& reference_model; // 参考模型（通常与policy_model初始相同）
    RewardModel& reward_model;
    AdamW optimizer;
    
    // PPO超参数
    scalar_t clip_epsilon = 0.2f;
    scalar_t kl_coef = 0.01f;
    scalar_t gamma = 0.99f;    // 折扣因子
    scalar_t lambda = 0.95f;   // GAE参数
    
public:
    PPO(GPTModel& policy, GPTModel& ref, RewardModel& rm)
        : policy_model(policy), reference_model(ref), reward_model(rm),
          optimizer(1e-6, 0.9, 0.999, 1e-8, 0.01) {}
    
    struct RolloutBuffer {
        std::vector<Tensor> states;      // 输入状态
        std::vector<Tensor> actions;     // 生成的tokens
        std::vector<scalar_t> rewards;   // 奖励
        std::vector<scalar_t> values;    // 价值估计
        std::vector<scalar_t> logprobs;  // 对数概率
        std::vector<bool> dones;         // 是否结束
    };
    
    RolloutBuffer collect_rollouts(const Tensor& initial_prompt, 
                                  size_t max_steps = 100) {
        RolloutBuffer buffer;
        
        Tensor current_state = initial_prompt;
        
        for (size_t step = 0; step < max_steps; step++) {
            // 使用当前策略生成下一个token
            Tensor action_dist = policy_model.forward(current_state);
            
            // 采样动作
            Tensor action = sample_action(action_dist);
            
            // 计算对数概率
            scalar_t logprob = compute_log_prob(action_dist, action);
            
            // 获取价值估计（简化）
            scalar_t value = estimate_value(current_state);
            
            // 执行动作（将token添加到序列）
            Tensor next_state = append_token(current_state, action);
            
            // 计算奖励
            scalar_t reward = compute_reward(next_state);
            
            // 检查是否结束
            bool done = (action[{0, 0}] == tokenizer->get_eos_id()) || 
                       (step == max_steps - 1);
            
            // 存储到缓冲区
            buffer.states.push_back(current_state);
            buffer.actions.push_back(action);
            buffer.rewards.push_back(reward);
            buffer.values.push_back(value);
            buffer.logprobs.push_back(logprob);
            buffer.dones.push_back(done);
            
            if (done) break;
            
            current_state = next_state;
        }
        
        return buffer;
    }
    
    void update_policy(const RolloutBuffer& buffer) {
        // 计算优势函数
        std::vector<scalar_t> advantages = compute_advantages(buffer);
        
        // 计算回报
        std::vector<scalar_t> returns = compute_returns(buffer);
        
        // PPO更新
        for (size_t i = 0; i < buffer.states.size(); i++) {
            // 获取新旧策略的概率比
            scalar_t ratio = compute_probability_ratio(buffer, i);
            
            // 裁剪的目标函数
            scalar_t surr1 = ratio * advantages[i];
            scalar_t surr2 = std::clamp(ratio, 1.0f - clip_epsilon, 
                                       1.0f + clip_epsilon) * advantages[i];
            
            scalar_t policy_loss = -std::min(surr1, surr2);
            
            // KL散度惩罚
            scalar_t kl_penalty = compute_kl_divergence(buffer, i);
            
            // 价值损失
            scalar_t value_loss = compute_value_loss(buffer, i, returns[i]);
            
            // 总损失
            scalar_t total_loss = policy_loss + kl_coef * kl_penalty + 
                                0.5f * value_loss;
            
            // 反向传播和优化
            optimizer.zero_grad();
            // compute_gradients(total_loss);
            optimizer.step();
        }
    }
    
private:
    Tensor sample_action(const Tensor& distribution) {
        // 从分布中采样一个token
        // 简化实现：取概率最大的token
        size_t vocab_size = distribution.shape[2];
        
        scalar_t max_prob = -std::numeric_limits<scalar_t>::max();
        token_id_t best_token = 0;
        
        for (size_t v = 0; v < vocab_size; v++) {
            if (distribution[{0, 0, v}] > max_prob) {
                max_prob = distribution[{0, 0, v}];
                best_token = v;
            }
        }
        
        Tensor action({1, 1});
        action[{0, 0}] = best_token;
        return action;
    }
    
    scalar_t compute_log_prob(const Tensor& distribution, const Tensor& action) {
        token_id_t token = static_cast<token_id_t>(action[{0, 0}]);
        return std::log(distribution[{0, 0, token}] + 1e-10f);
    }
    
    scalar_t estimate_value(const Tensor& state) {
        // 简化：使用奖励模型的输出作为价值估计
        Tensor dummy_mask(state.shape);
        std::fill(dummy_mask.data.begin(), dummy_mask.data.end(), 1.0f);
        
        return reward_model.forward(state, dummy_mask);
    }
    
    scalar_t compute_reward(const Tensor& state) {
        // 使用奖励模型计算即时奖励
        Tensor dummy_mask(state.shape);
        std::fill(dummy_mask.data.begin(), dummy_mask.data.end(), 1.0f);
        
        scalar_t reward = reward_model.forward(state, dummy_mask);
        
        // KL惩罚（防止偏离参考模型太远）
        scalar_t kl_penalty = compute_kl_penalty(state);
        
        return reward - kl_coef * kl_penalty;
    }
    
    scalar_t compute_kl_penalty(const Tensor& state) {
        // 计算当前策略和参考策略之间的KL散度
        // 简化实现
        return 0.1f;
    }
    
    std::vector<scalar_t> compute_advantages(const RolloutBuffer& buffer) {
        // 使用GAE（Generalized Advantage Estimation）计算优势函数
        size_t n = buffer.rewards.size();
        std::vector<scalar_t> advantages(n, 0.0f);
        
        scalar_t last_advantage = 0.0f;
        for (int t = n - 1; t >= 0; t--) {
            scalar_t delta = buffer.rewards[t] + 
                           gamma * (t < n - 1 ? buffer.values[t + 1] : 0.0f) - 
                           buffer.values[t];
            
            advantages[t] = delta + gamma * lambda * 
                          (t < n - 1 ? (1.0f - buffer.dones[t]) : 0.0f) * 
                          last_advantage;
            
            last_advantage = advantages[t];
        }
        
        return advantages;
    }
    
    std::vector<scalar_t> compute_returns(const RolloutBuffer& buffer) {
        // 计算折扣回报
        size_t n = buffer.rewards.size();
        std::vector<scalar_t> returns(n, 0.0f);
        
        scalar_t last_return = 0.0f;
        for (int t = n - 1; t >= 0; t--) {
            returns[t] = buffer.rewards[t] + gamma * 
                        (t < n - 1 ? (1.0f - buffer.dones[t]) : 0.0f) * 
                        last_return;
            last_return = returns[t];
        }
        
        return returns;
    }
    
    scalar_t compute_probability_ratio(const RolloutBuffer& buffer, size_t idx) {
        // 计算新旧策略的概率比
        // 在实际实现中，需要重新计算当前策略的概率
        return 1.0f;
    }
    
    scalar_t compute_kl_divergence(const RolloutBuffer& buffer, size_t idx) {
        // 计算KL散度
        return 0.0f;
    }
    
    scalar_t compute_value_loss(const RolloutBuffer& buffer, 
                               size_t idx, scalar_t target_return) {
        scalar_t value_pred = buffer.values[idx];
        scalar_t error = target_return - value_pred;
        return 0.5f * error * error;
    }
    
    Tensor append_token(const Tensor& state, const Tensor& token) {
        // 将token添加到序列末尾
        size_t batch_size = state.shape[0];
        size_t seq_len = state.shape[1];
        
        Tensor new_state({batch_size, seq_len + 1});
        
        // 复制原有序列
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                new_state[{b, s}] = state[{b, s}];
            }
            // 添加新token
            new_state[{b, seq_len}] = token[{b, 0}];
        }
        
        return new_state;
    }
};

#endif // RLHF_TRAINING_HPP
```

## 8. 完整的训练管道整合

```cpp
// complete_pipeline.cpp
#include <iostream>
#include <memory>
#include <vector>

#include "pretraining.hpp"
#include "train_pipeline.hpp"
#include "sft_training.hpp"
#include "rlhf_training.hpp"

class CompleteLLMTrainingPipeline {
private:
    // 配置
    struct PipelineConfig {
        ModelConfig model_config;
        
        struct {
            size_t total_steps = 100000;
            std::string data_dir = "./data/pretrain";
        } pretrain;
        
        struct {
            size_t epochs = 3;
            std::string sft_data = "./data/sft.json";
        } sft;
        
        struct {
            size_t preference_samples = 10000;
            std::string preference_data = "./data/preferences.json";
        } reward_model;
        
        struct {
            size_t ppo_epochs = 5;
            size_t rollout_steps = 100;
        } rlhf;
    };
    
    PipelineConfig config;
    
    // 组件
    std::unique_ptr<GPTModel> model;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<DataPipeline> data_pipeline;
    std::unique_ptr<TrainingMonitor> monitor;
    
    // 训练状态
    enum PipelineStage {
        STAGE_PRETRAIN,
        STAGE_SFT,
        STAGE_REWARD_MODEL,
        STAGE_RLHF,
        STAGE_DONE
    };
    
    PipelineStage current_stage = STAGE_PRETRAIN;
    
public:
    CompleteLLMTrainingPipeline(const PipelineConfig& cfg) : config(cfg) {
        initialize();
    }
    
    void initialize() {
        std::cout << "初始化LLM训练管道..." << std::endl;
        
        // 1. 初始化分词器
        tokenizer = std::make_unique<Tokenizer>(config.model_config.vocab_size);
        
        // 2. 初始化模型
        model = std::make_unique<GPTModel>(config.model_config);
        
        std::cout << "模型初始化完成，总参数量: " 
                  << config.model_config.total_params() << std::endl;
    }
    
    void run_pipeline() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "开始完整LLM训练管道" << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;
        
        // 阶段1: 预训练
        run_pretraining();
        
        // 阶段2: 有监督微调
        run_sft();
        
        // 阶段3: 奖励建模
        run_reward_model_training();
        
        // 阶段4: RLHF
        run_rlhf();
        
        // 阶段5: 最终评估和保存
        finalize();
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "LLM训练管道完成!" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
    
private:
    void run_pretraining() {
        std::cout << "\n阶段1: 预训练 (Pre-training)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // 创建数据管道
        data_pipeline = std::make_unique<DataPipeline>(
            config.pretrain.data_dir, tokenizer.get());
        
        // 创建训练器
        PreTrainingPipeline pretrainer(
            config.model_config,
            config.pretrain.data_dir,
            config.pretrain.total_steps
        );
        
        // 开始训练
        pretrainer.train();
        
        std::cout << "预训练完成!" << std::endl;
        current_stage = STAGE_SFT;
    }
    
    void run_sft() {
        std::cout << "\n阶段2: 有监督微调 (Supervised Fine-tuning)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // 创建SFT训练器
        SFTTrainer sft_trainer(
            *model,
            tokenizer.get(),
            config.sft.epochs
        );
        
        // 开始SFT训练
        sft_trainer.train();
        
        std::cout << "SFT训练完成!" << std::endl;
        current_stage = STAGE_REWARD_MODEL;
    }
    
    void run_reward_model_training() {
        std::cout << "\n阶段3: 奖励模型训练 (Reward Modeling)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // 创建奖励模型（基于SFT后的模型）
        RewardModel reward_model(*model);
        
        // 创建奖励模型训练器
        RewardModelTrainer rm_trainer(reward_model, tokenizer.get());
        
        // 训练奖励模型
        std::cout << "训练奖励模型..." << std::endl;
        for (size_t step = 0; step < 10000; step++) {
            rm_trainer.train_step();
            
            if (step % 1000 == 0) {
                std::cout << "奖励模型训练步数: " << step << "/10000" << std::endl;
            }
        }
        
        std::cout << "奖励模型训练完成!" << std::endl;
        current_stage = STAGE_RLHF;
    }
    
    void run_rlhf() {
        std::cout << "\n阶段4: RLHF (Reinforcement Learning from Human Feedback)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // 创建参考模型（复制当前模型）
        GPTModel reference_model = *model;  // 需要深拷贝
        
        // 创建奖励模型
        RewardModel reward_model(*model);
        
        // 创建PPO训练器
        PPO ppo_trainer(*model, reference_model, reward_model);
        
        std::cout << "开始PPO训练..." << std::endl;
        
        // 训练循环
        for (size_t epoch = 0; epoch < config.rlhf.ppo_epochs; epoch++) {
            std::cout << "PPO Epoch " << (epoch + 1) << "/" 
                      << config.rlhf.ppo_epochs << std::endl;
            
            // 收集经验
            Tensor initial_prompt({1, 10});  // 示例提示
            // 填充提示...
            
            PPO::RolloutBuffer buffer = ppo_trainer.collect_rollouts(
                initial_prompt, config.rlhf.rollout_steps);
            
            // 更新策略
            ppo_trainer.update_policy(buffer);
            
            // 评估当前策略
            evaluate_policy(epoch);
        }
        
        std::cout << "RLHF训练完成!" << std::endl;
        current_stage = STAGE_DONE;
    }
    
    void evaluate_policy(size_t epoch) {
        // 评估当前策略的性能
        std::cout << "评估策略..." << std::endl;
        
        // 示例：生成一些文本并评估质量
        std::vector<std::string> test_prompts = {
            "解释什么是人工智能",
            "写一个关于友谊的短故事",
            "如何学习编程",
            "描述巴黎的风景"
        };
        
        for (const auto& prompt : test_prompts) {
            std::vector<token_id_t> tokens = tokenizer->encode(prompt);
            Tensor input_tensor({1, (size_t)tokens.size()});
            
            for (size_t i = 0; i < tokens.size(); i++) {
                input_tensor[{0, i}] = tokens[i];
            }
            
            // 生成文本
            Tensor generated = generate_text(input_tensor, 50);
            
            // 解码并显示
            std::vector<token_id_t> output_tokens(generated.shape[1]);
            for (size_t i = 0; i < output_tokens.size(); i++) {
                output_tokens[i] = static_cast<token_id_t>(generated[{0, i}]);
            }
            
            std::string generated_text = tokenizer->decode(output_tokens);
            
            std::cout << "Prompt: " << prompt << std::endl;
            std::cout << "Generated: " << generated_text.substr(0, 100) 
                      << "..." << std::endl;
            std::cout << std::endl;
        }
    }
    
    Tensor generate_text(const Tensor& input, size_t max_length) {
        // 简化的文本生成
        Tensor output = input;
        
        for (size_t i = 0; i < max_length; i++) {
            // 前向传播
            Tensor attention_mask(output.shape);
            std::fill(attention_mask.data.begin(), attention_mask.data.end(), 1.0f);
            
            Tensor logits = model->forward(output, attention_mask);
            
            // 取最后一个token的预测
            size_t last_idx = output.shape[1] - 1;
            
            // 找到概率最高的token
            size_t vocab_size = logits.shape[2];
            scalar_t max_prob = -std::numeric_limits<scalar_t>::max();
            token_id_t next_token = 0;
            
            for (size_t v = 0; v < vocab_size; v++) {
                if (logits[{0, last_idx, v}] > max_prob) {
                    max_prob = logits[{0, last_idx, v}];
                    next_token = v;
                }
            }
            
            // 添加新token到输出
            Tensor new_output({output.shape[0], output.shape[1] + 1});
            
            // 复制原有tokens
            for (size_t s = 0; s < output.shape[1]; s++) {
                new_output[{0, s}] = output[{0, s}];
            }
            
            // 添加新token
            new_output[{0, output.shape[1]}] = next_token;
            
            output = new_output;
            
            // 如果生成EOS token，停止
            if (next_token == tokenizer->get_eos_id()) {
                break;
            }
        }
        
        return output;
    }
    
    void finalize() {
        std::cout << "\n阶段5: 最终处理和保存" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // 保存最终模型
        save_final_model();
        
        // 生成最终评估报告
        generate_evaluation_report();
        
        std::cout << "管道完成，模型已保存!" << std::endl;
    }
    
    void save_final_model() {
        std::string model_path = "./final_model/model.bin";
        std::string config_path = "./final_model/config.json";
        
        std::cout << "保存模型到: " << model_path << std::endl;
        
        // 在实际实现中，这里需要序列化模型的所有参数
        // 保存配置
        std::ofstream config_file(config_path);
        config_file << "{\n"
                   << "  \"vocab_size\": " << config.model_config.vocab_size << ",\n"
                   << "  \"hidden_size\": " << config.model_config.hidden_size << ",\n"
                   << "  \"num_layers\": " << config.model_config.num_layers << ",\n"
                   << "  \"num_heads\": " << config.model_config.num_heads << ",\n"
                   << "  \"max_seq_len\": " << config.model_config.max_seq_len << ",\n"
                   << "  \"ffn_dim\": " << config.model_config.ffn_dim << "\n"
                   << "}" << std::endl;
        config_file.close();
    }
    
    void generate_evaluation_report() {
        std::ofstream report("./final_model/evaluation_report.txt");
        
        report << "LLM训练管道最终报告\n";
        report << "====================\n\n";
        
        report << "模型配置:\n";
        report << "- 词汇表大小: " << config.model_config.vocab_size << "\n";
        report << "- 隐藏层维度: " << config.model_config.hidden_size << "\n";
        report << "- 层数: " << config.model_config.num_layers << "\n";
        report << "- 注意力头数: " << config.model_config.num_heads << "\n";
        report << "- 总参数量: " << config.model_config.total_params() << "\n\n";
        
        report << "训练流程:\n";
        report << "- 预训练步数: " << config.pretrain.total_steps << "\n";
        report << "- SFT轮数: " << config.sft.epochs << "\n";
        report << "- RLHF轮数: " << config.rlhf.ppo_epochs << "\n\n";
        
        report << "生成示例:\n";
        
        // 生成一些示例文本
        std::vector<std::string> prompts = {
            "人工智能的未来是",
            "机器学习可以帮助我们",
            "写一首短诗:"
        };
        
        for (const auto& prompt : prompts) {
            report << "Prompt: " << prompt << "\n";
            
            // 编码提示
            std::vector<token_id_t> tokens = tokenizer->encode(prompt);
            Tensor input({1, (size_t)tokens.size()});
            
            for (size_t i = 0; i < tokens.size(); i++) {
                input[{0, i}] = tokens[i];
            }
            
            // 生成文本
            Tensor generated = generate_text(input, 50);
            
            // 解码
            std::vector<token_id_t> output_tokens(generated.shape[1]);
            for (size_t i = 0; i < output_tokens.size(); i++) {
                output_tokens[i] = static_cast<token_id_t>(generated[{0, i}]);
            }
            
            std::string generated_text = tokenizer->decode(output_tokens);
            report << "Generated: " << generated_text << "\n\n";
        }
        
        report.close();
    }
};

int main() {
    std::cout << "LLM训练管道模拟器" << std::endl;
    std::cout << "=================\n" << std::endl;
    
    // 配置小模型用于演示
    CompleteLLMTrainingPipeline::PipelineConfig config;
    
    config.model_config.vocab_size = 50257;
    config.model_config.hidden_size = 768;
    config.model_config.num_layers = 12;
    config.model_config.num_heads = 12;
    config.model_config.max_seq_len = 1024;
    config.model_config.ffn_dim = 3072;
    
    config.pretrain.total_steps = 1000;  // 小规模演示
    config.sft.epochs = 1;
    config.rlhf.ppo_epochs = 2;
    
    // 创建并运行管道
    CompleteLLMTrainingPipeline pipeline(config);
    
    try {
        pipeline.run_pipeline();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## 9. 训练流程可视化

```
完整LLM训练管道流程图
================================

初始化
  │
  ├─▶ 数据预处理
  │     │
  │     ├─▶ 文本清洗
  │     ├─▶ 分词
  │     └─▶ 创建训练批次
  │
  ├─▶ 预训练阶段 (100,000步)
  │     │
  │     ├─▶ 前向传播: token → 嵌入 → Transformer → logits
  │     ├─▶ 损失计算: 交叉熵损失 (语言建模)
  │     ├─▶ 反向传播: 计算梯度
  │     ├─▶ 优化器步骤: AdamW更新权重
  │     └─▶ 定期评估和保存检查点
  │
  ├─▶ 有监督微调 (3轮)
  │     │
  │     ├─▶ 加载指令-回复对数据
  │     ├─▶ 只计算assistant部分的损失
  │     ├─▶ 微调全部参数
  │     └─▶ 保存SFT模型
  │
  ├─▶ 奖励建模
  │     │
  │     ├─▶ 收集偏好数据 (chosen vs rejected)
  │     ├─▶ 在SFT模型基础上添加奖励头
  │     ├─▶ 训练奖励模型: 成对排名损失
  │     └─▶ 保存奖励模型
  │
  ├─▶ RLHF训练 (5轮PPO)
  │     │
  │     ├─▶ 收集经验: 当前策略生成回复
  │     ├─▶ 计算奖励: 奖励模型 + KL惩罚
  │     ├─▶ PPO更新: 策略梯度 + 价值函数更新
  │     └─▶ 定期评估策略性能
  │
  └─▶ 最终处理和部署
        │
        ├─▶ 模型量化 (可选)
        ├─▶ 生成评估报告
        └─▶ 导出为可部署格式
```

## 10. 关键算法总结

### 预训练关键算法：
1. **自回归语言建模**: $P(w_t|w_{<t})$
2. **Transformer前向传播**: $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
3. **AdamW优化器**: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$, $\theta_t = \theta_{t-1} - \eta(\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon) + \lambda\theta_{t-1})$
4. **梯度裁剪**: $g \leftarrow g \times \min(1, \frac{\text{max\_norm}}{\|g\|_2})$

### SFT关键算法：
1. **指令微调**: 最小化 $-\sum \log P(\text{response}|\text{instruction})$
2. **只计算assistant部分损失**: 掩码掉user和system部分的损失

### 奖励建模关键算法：
1. **Bradley-Terry模型**: $P(y_w \succ y_l|x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}$
2. **成对排名损失**: $\mathcal{L} = -\mathbb{E}_{(x,y_w,y_l)}[\log\sigma(r(x,y_w) - r(x,y_l))]$

### RLHF关键算法：
1. **PPO目标函数**: $L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$
2. **GAE优势估计**: $\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$
3. **KL惩罚**: $\mathcal{L}^{KL} = \beta \cdot \text{KL}[\pi_\theta||\pi_{\text{ref}}]$

## 学习建议：

1. **纸上模拟**: 拿一张纸，画出张量形状的流动
   - 输入: [batch=8, seq=512]
   - 嵌入后: [8, 512, 768]
   - 注意力后: 形状不变
   - 输出logits: [8, 512, 50257]

2. **逐步实现**:
   - 第1天: 实现Tensor类和基础运算
   - 第2-3天: 实现Transformer层
   - 第4天: 实现优化器和损失函数
   - 第5天: 整合成完整训练循环

3. **调试技巧**:
   - 从小模型开始 (hidden_size=64)
   - 使用固定数据验证前向传播
   - 检查梯度是否合理 (不是NaN或过大)
   - 逐步增加复杂度

这个完整的C++伪代码实现展示了LLM训练管道的所有核心组件。虽然为了可读性进行了简化，但它包含了理解整个流程所需的所有关键概念和算法。你可以用它作为蓝图，在实际实现时填充细节。
