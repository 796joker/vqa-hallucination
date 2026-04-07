# 项目状态总览

> 最后更新: 2026-04-06 18:30

## 一、项目概况

| 项目 | 详情 |
|------|------|
| 课程 | 大模型后训练技术 (40% 总成绩) |
| 方向 | 方向6: 视觉问答模型 |
| 基座模型 | Qwen3-VL-8B-Instruct |
| SFT 数据 | LLaVA-Instruct-150K (5K-50K 消融) |
| DPO 数据 | RLHF-V (5,733 pairs, 人工标注) |
| 评估基准 | POPE + CHAIR + MME |
| 消融规模 | 5 维度, 20 组模型配置 |
| GPU 时长 | ~100 GPU-hours on A100-80GB |

---

## 二、已完成的工作

### 实验 (全部在 236 服务器: ssh research@115.190.215.236)
- [x] 20 组模型训练 (8 SFT + 12 DPO)
- [x] POPE 评估: 20 模型 × 3 splits = 60 组
- [x] CHAIR 评估: 17 模型
- [x] MME 评估: 4 关键模型
- [x] Case study: 40 案例 (10 图 × 4 问 × 3 模型)
- [x] 数据全量验证 → DATA_VERIFICATION.md

### 文档
- [x] 课程报告 (`课程报告.md`) — 4-6 页, 含 Q&A 准备, 数据已校验
- [x] 实验数据验证 (`DATA_VERIFICATION.md`) — 已推送至 GitHub
- [x] 技术报告 (`TECHNICAL_REPORT.md`) — 25000 字完整版 (备用)
- [x] 修复计划 (`FIX_PLAN.md`) — RLAIF-V/MMBench 备选方案

### 代码 & 仓库
- [x] GitHub: https://github.com/796joker/vqa-hallucination
- [x] 2979 文件已提交, working tree clean
- [x] 包含全部 adapter 权重 (~87MB × 18), 评估结果, 配置文件
- [x] RLHF-V 数据集已加入仓库 (data/dpo_data/rlhf-v.parquet, 348MB)

### Gradio Demo (`demo/app.py`)
- [x] Tab 1: 对比演示 — Base vs True Optimal 并排回答 + 差异分析
- [x] Tab 2: 幻觉检测 (POPE) — 8 物体批量 yes/no 测试
- [x] Tab 3: 描述对比 (CHAIR) — 图像描述生成对比
- [x] Tab 4: 实验指标 — 核心数据表 + 四大发现
- [x] Tab 5: 消融实验 — 五维度消融结果
- [x] Tab 6: 关于 — 项目信息 + 环境配置

### 可视化 (`report/figures/`)
- [x] 16 张专业图表 (POPE/CHAIR/MME/消融/文献对比)
- [x] 6 张案例研究图
- [x] 幻觉维度热力图 (`results/figures/`)

---

## 三、课程要求差异说明

| 项目 | 课程推荐 | 我们使用 | 理由 |
|------|---------|---------|------|
| 模型 | Qwen3-VL-2B | **8B** | 更强基线, LoRA 仅 0.29% 参数, 训练时间符合预期 |
| DPO 数据 | RLAIF-V (83K, AI标注) | **RLHF-V** (5.7K, 人工标注) | 同一团队, 人工标注质量更高 |
| 评估 | MMBench | **MME** | 幻觉研究更主流, 14 子任务支持细粒度分析 |

> 详细解释见课程报告 Section 1.3 和附录 G Q&A 准备。

---

## 四、备选数据 (已下载, 未使用)

| 数据集 | 位置 (236 服务器) | 大小 | 用途 |
|--------|-----------------|------|------|
| RLAIF-V | /mnt/disk2/lijunlin/downloads/datasets/RLAIF-V/ | 12GB | 如需重跑 DPO |
| MMBench | /mnt/disk2/lijunlin/downloads/datasets/MMBench/ | 503MB | 如需补跑评估 |

---

## 五、课程展示准备清单

### 报告 (20%) — 已完成
- [x] 课程报告 (`课程报告.md`): 摘要→引言→方法→实验→分析→附录
- [x] 实验设计思想说明 (Section 2.1)
- [x] 服务器环境详情 (附录 B)
- [x] 数据集选型说明 (Section 1.3)
- [x] 假设验证/推翻表 (Section 4.1)
- [x] 课程理论联系 (Section 4.2)
- [x] 踩坑经验 (Section 4.3, 含 think 标签和 DeepSpeed 问题)
- [x] 前沿论文对比 (附录 D)
- [x] Q&A 准备 (附录 G, 12 个问题)

### 展示 (20%) — 部分完成
- [x] Gradio Demo 代码 (`demo/app.py`, 6 Tabs)
- [ ] **在服务器上实际运行 Demo** (需要 ~32GB 空闲 GPU)
- [ ] **录制备份视频** (防止现场 Demo 失败, -3~5 分扣分)
- [ ] **PPT 制作** (8-10 页: 问题定义→架构图→数据训练→Demo→结果→发现→反思)
- [ ] **排练计时** (严格 15 分钟, 超时 >18min 扣 2-3 分)

### 加分项 (最多 +5)
- [x] 创新方法组合 (+1-2): 5 维消融 + 六维幻觉分析
- [x] 高质量可视化 (+1): 22 张图表
- [x] 深入失败分析 (+1): IPO 崩溃, 低 beta 崩溃, SFT 悖论
- [x] 前沿论文对比 (+1): 与 HA-DPO/RLHF-V/VCD 对比
- [x] 可复用代码 (+1): GitHub 仓库完整

---

## 六、服务器信息

### 236 服务器 (主实验服务器)
- SSH: `ssh research@115.190.215.236`
- 项目: `/mnt/disk2/lijunlin/vqa-hallucination`
- Conda: `/mnt/disk3/conda/miniconda3/envs/zh/bin`
- RLHF-V: `/mnt/disk2/lijunlin/hf_cache/rlhf_v_data/rlhf-v.parquet` (现已复制到项目内)
- 模型: `../downloads/models/Qwen3-VL-8B-Instruct`
- COCO: `../downloads/coco/val2014`

### 37 服务器 (不完整副本, 仅备用)
- SSH: `ssh research@115.190.234.37`
- 需要 Clash 代理绕过 (DIRECT 规则)
- 项目: `/mnt/disk2/lijunlin/vqa-hallucination` (只有基础 SFT+DPO)

---

## 七、RLAIF-V 补充实验 (进行中)

**启动时间**: 2026-04-06 18:32

| 实验 | GPU | 进度 | 预计完成 |
|------|-----|------|---------|
| baseline (SFT50K+β=0.1+3ep) | 3,4 DDP | 2.8% | 4/7 ~08:00 |
| optimal (SFT5K+β=1.0+1ep) | 5,6 DDP | 8.4% | 4/6 ~23:00 |
| epoch1 (SFT50K+β=0.1+1ep) | 0,1 DDP | 2.2% | 4/6 ~23:00 |
| only (Base+β=0.1+3ep) | 7 单卡 | tokenize 91% | 4/7 ~12:00 |

详见 `RLAIFV_MMBENCH_PLAN.md`

---

## 八、待做事项 (按优先级)

1. **监控训练** — 5 分钟自动检查 GPU + 进度 + 错误 (已配置)
2. **训练完成后评估** — POPE + CHAIR (已有脚本), MMBench (待写脚本)
3. **运行 Demo** — 在有空闲 GPU 时运行 `bash scripts/run_demo.sh`
4. **录制备份视频** — Demo 运行时录屏 2-3 个成功案例 + 1 个失败案例
5. **制作 PPT** — 8-10 页, 从课程报告提取关键内容
6. **排练** — 控制在 15 分钟内, 注意 5+5+3+2 时间分配
7. **更新课程报告** — 加入 RLAIF-V 对比结果 (训练+评估完成后)
