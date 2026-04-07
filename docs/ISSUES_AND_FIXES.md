# 实验过程问题排查与修复记录

> 更新时间: 2026-03-29

## 问题 1（严重）: eval_pope.py 无法正确解析 DPO 模型回答

### 现象
SFT+DPO 模型 POPE 评估结果异常：
- Accuracy ≈ 0.526, F1 ≈ 0.098, Yes Ratio = 2.6%
- 看似模型完全崩溃，几乎全部回答 "No"

### 根因
Qwen3-VL 模型经过 SFT+DPO 后输出格式发生变化：
- **Base 模型**: 无 `<think>` 标签，直接输出 "yes"/"no"
- **SFT 模型**: 输出 `<think>\n\n</think>\n\nYes/No...`（正常配对标签）
- **SFT+DPO 模型**: 输出 `<think>\n\n<think>\n\nYes/No...`（**双重开标签，无闭合标签**，91.6%）
- **DPO-only 模型**: 无 `<think>` 标签

原 `parse_yesno` 使用 `re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)` 需要配对标签，
对 `<think>...<think>` 格式无法匹配，残留 `<think>` 导致所有回答被默认解析为 "no"。

### 修复
```python
# 修复前（需要配对标签）
text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().lower()

# 修复后（删除所有 think 标签，无论配对与否）
text = re.sub(r"</?think>", "", text).strip().lower()
```

### 验证
- 修复后 SFT+DPO POPE Accuracy: 0.526 → **0.819**，Yes Ratio: 2.6% → **32.2%**
- Base/SFT/DPO-only 模型结果完全不受影响
- 修复后的 Yes Ratio 32.2% 偏低（理想 50%），反映 DPO 使模型更保守（合理的实验发现）

### 受影响文件
- `eval/eval_pope.py` — `parse_yesno()` 函数
- `eval/eval_chair.py` — `caption_to_objects()` 函数（同样的正则，但对 CHAIR 结果影响极小）

### 影响范围
- 需重新评分：所有已完成的 POPE 评估（已完成 ✅）
- 不需重跑：训练、答案生成、正在进行的 POPE eval（答案数据正确，评分脚本已修复）

---

## 问题 2（中等）: CHAIR 评分依赖 nltk wordnet 但服务器未安装

### 现象
运行 `eval_chair.py` 时卡住不动

### 根因
`eval_chair.py` 使用 `WordNetLemmatizer` 需要 nltk wordnet 数据。
代码中有自动下载逻辑，但服务器网络不稳定导致卡住。

### 修复
手动在服务器执行：
```python
import nltk
nltk.download('wordnet', quiet=True)
```
（2026-03-29 成功下载到 /home/research/nltk_data/corpora/wordnet.zip）

---

## 问题 3（轻微）: SFT+DPO 模型 7.8% 回答不以 yes/no 开头

### 现象
SFT+DPO 模型在 POPE 中有 233/3000 (7.8%) 的回答不以 "Yes"/"No" 开头，
而是间接回答如 "Based on the image, there is no..."、"There are people in the image..."

### 分析
- 这是 **DPO 训练导致的模型行为变化**，不是解析 bug
- 其他模型（Base、SFT、DPO-only）均为 0% 间接回答
- 使用 `suppress_tokens=[151667]` 抑制 `<think>` token **不能修复此问题**
- 更复杂的解析器（检测 "there is/are" 模式）可提升 ~2.7pp 准确率
- 但 POPE 标准评估协议就是检查回答是否以 yes/no 开头，间接回答按 "no" 处理是惯例

### 决策
保持简单解析器（与文献一致），将 7.8% 间接回答视为 DPO 模型的缺陷（过度分析、不遵循指令格式），
这本身就是一个值得在论文中讨论的发现。

---

## 问题 4（参考）: Qwen3-VL thinking mode 与微调的交互

### 背景
Qwen3-VL-8B-Instruct 的 `<think>` (token 151667) 和 `</think>` (token 151668) 是特殊 token。
模型的 thinking 行为是训练习得的，不是 chat template 控制的。

### 训练 template 信息
所有训练 config 使用 `template: qwen3_vl`（LLaMA Factory 的 Qwen3-VL 默认 template），
该 template 不禁用 thinking mode。LLaMA Factory 也提供 `qwen3_vl_nothink` template。
训练数据（LLaVA SFT, RLHF-V DPO）本身不含 `<think>` 标签，thinking 行为是模型固有的。

### 微调后的行为变化
| 训练方式 | think 标签行为 | 原因推测 |
|----------|-------------|---------|
| Base（无微调） | 不产生 think 标签 | greedy decode + max_tokens=32 抑制 |
| SFT only | `<think>...</think>` 正常配对 | SFT 保留了 base 的 thinking 能力 |
| SFT+DPO | `<think>...<think>` 畸形 | DPO 改变了 token 分布，破坏 think 配对 |
| DPO-only (无 SFT) | 不产生 think 标签 | 跳过 SFT，thinking 行为未被激活 |

### 建议（未来实验）
- 可在生成脚本中添加 `suppress_tokens=[151667, 151668]` 抑制 thinking
- 或训练时改用 `template: qwen3_vl_nothink` 从源头禁用
- 对当前实验不是必须的，eval 脚本已修复可正确处理

---

## 修复后的正确 POPE 结果

| 模型 | Random Acc | Random F1 | Pop. Acc | Pop. F1 | Adv. Acc | Adv. F1 | Yes Ratio |
|------|-----------|----------|---------|--------|---------|--------|-----------|
| Base | 0.912 | 0.906 | 0.889 | 0.883 | 0.870 | 0.866 | 43.1% |
| SFT | 0.899 | 0.895 | 0.854 | 0.856 | 0.814 | 0.823 | 46.9% |
| SFT+DPO | 0.819 | 0.780 | 0.815 | 0.776 | 0.810 | 0.772 | 32.2% |
| DPO-only | 0.908 | 0.900 | 0.886 | 0.880 | 0.873 | 0.868 | 42.6% |

## 修复后的 CHAIR 结果（无变化）

| 模型 | CHAIR_s (↓) | CHAIR_i (↓) | Recall (↑) |
|------|------------|------------|-----------|
| Base | 65.73% | 33.31% | 81.37% |
| SFT | 31.25% | 16.64% | 64.89% |
| SFT+DPO | 39.31% | 18.88% | 77.27% |
