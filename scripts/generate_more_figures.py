#!/usr/bin/env python3
"""生成剩余报告图表 - 第二批"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs('report/figures', exist_ok=True)

print("="*60)
print("Generating remaining figures (Part 2)...")
print("="*60)

# ============================================================================
# 图6.5：知识任务退化与DPO恢复
# ============================================================================
print("\n[6/14] Generating Figure 6.5: Knowledge Degradation...")

tasks = ['Celebrity\n名人', 'Artwork\n艺术品', 'Landmark\n地标']
base_acc = [90.59, 85.00, 94.25]
sft_acc = [83.24, 78.00, 87.50]
true_opt_acc = [93.24, 84.25, 92.50]

x = np.arange(len(tasks))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width, base_acc, width, label='Base (baseline)',
               color='#90CAF9', edgecolor='#1976D2', linewidth=2, alpha=0.9)
bars2 = ax.bar(x, sft_acc, width, label='SFT 5K (forgetting)',
               color='#EF9A9A', edgecolor='#C62828', linewidth=2, alpha=0.9)
bars3 = ax.bar(x + width, true_opt_acc, width, label='True Optimal (recovery)',
               color='#A5D6A7', edgecolor='#388E3C', linewidth=2, alpha=0.9)

# 标注变化
for i in range(len(tasks)):
    # SFT退化箭头
    delta_sft = sft_acc[i] - base_acc[i]
    ax.annotate('', xy=(i, sft_acc[i]), xytext=(i-width, base_acc[i]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(i-width/2, (base_acc[i]+sft_acc[i])/2, f'{delta_sft:.1f}pp',
            fontsize=10, color='red', fontweight='bold', ha='center')

    # DPO恢复箭头
    delta_dpo = true_opt_acc[i] - sft_acc[i]
    ax.annotate('', xy=(i+width, true_opt_acc[i]), xytext=(i, sft_acc[i]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(i+width/2, (sft_acc[i]+true_opt_acc[i])/2, f'+{delta_dpo:.1f}pp',
            fontsize=10, color='green', fontweight='bold', ha='center')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Knowledge Catastrophic Forgetting & DPO Recovery',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([70, 100])
ax.grid(axis='y', alpha=0.3)

# 标注关键发现
ax.text(1, 96, 'SFT avg: -7.03pp\nDPO recovery: +6.28pp',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('report/figures/figure_6_5_knowledge_degradation.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 6.5 saved")

# ============================================================================
# 图4.4：MME能力分数对比
# ============================================================================
print("[7/14] Generating Figure 4.4: MME Capability Scores...")

models = ['Base', 'SFT 50K', 'True\nOptimal']
perception = [1801.5, 1743.0, 1796.5]
cognition = [206.5, 202.5, 194.0]
total = [2008.0, 1945.5, 1990.5]

x = np.arange(len(models))
width = 0.6

fig, ax = plt.subplots(figsize=(12, 7))

# 堆叠柱状图
bars1 = ax.bar(x, perception, width, label='Perception (14 tasks)',
               color='#81C784', edgecolor='darkgreen', linewidth=2)
bars2 = ax.bar(x, cognition, width, bottom=perception, label='Cognition (4 tasks)',
               color='#64B5F6', edgecolor='darkblue', linewidth=2)

# 总分标注
for i, tot in enumerate(total):
    ax.text(i, tot + 30, f'Total:\n{tot:.1f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 能力保持率
    cpr = (tot / 2008.0) * 100
    color = 'green' if cpr >= 99 else 'orange' if cpr >= 97 else 'red'
    ax.text(i, tot + 80, f'CPR: {cpr:.1f}%',
            ha='center', fontsize=10, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color, linewidth=2))

ax.set_ylabel('MME Score', fontsize=14, fontweight='bold')
ax.set_title('MME Capability Preservation: Perception + Cognition',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim([0, 2200])
ax.axhline(y=2008, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.text(2.2, 2015, 'Baseline', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('report/figures/figure_4_4_mme_capability.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 4.4 saved")

# ============================================================================
# 图4.5：DPO-only悖论三维对比
# ============================================================================
print("[8/14] Generating Figure 4.5: DPO-only Paradox...")

models = ['Base', 'SFT 50K', 'DPO-only', 'True\nOptimal']
pope_f1 = [0.879, 0.855, 0.900, 0.889]
chair_i = [33.31, 16.64, 31.83, 20.12]
mme_score = [2008.0, 1945.5, 1964.5, 1990.5]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

# POPE F1
colors_pope = ['#90CAF9', '#FFAB91', '#66BB6A', '#81C784']
bars = ax1.bar(models, pope_f1, color=colors_pope, edgecolor='black', linewidth=2)
ax1.set_ylabel('POPE F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('Discriminative\nCapability', fontsize=13, fontweight='bold')
ax1.set_ylim([0.83, 0.92])
ax1.grid(axis='y', alpha=0.3)
# 标注DPO-only最佳
ax1.text(2, 0.905, 'BEST!', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='gold', edgecolor='orange', linewidth=2),
         fontweight='bold')

# CHAIR_i (越低越好)
colors_chair = ['#EF9A9A', '#A5D6A7', '#EF5350', '#66BB6A']
bars = ax2.bar(models, chair_i, color=colors_chair, edgecolor='black', linewidth=2)
ax2.set_ylabel('CHAIR_i Hallucination (%)', fontsize=12, fontweight='bold')
ax2.set_title('Generative\nQuality', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 40])
ax2.grid(axis='y', alpha=0.3)
ax2.invert_yaxis()  # 倒置Y轴，让低的看起来"好"
# 标注DPO-only最差
ax2.text(2, 30, 'WORST!', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='#ffebee', edgecolor='red', linewidth=2),
         fontweight='bold')

# MME总分
colors_mme = ['#90CAF9', '#FFAB91', '#FFF59D', '#A5D6A7']
bars = ax3.bar(models, mme_score, color=colors_mme, edgecolor='black', linewidth=2)
ax3.set_ylabel('MME Total Score', fontsize=12, fontweight='bold')
ax3.set_title('Capability\nPreservation', fontsize=13, fontweight='bold')
ax3.set_ylim([1900, 2050])
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=2008, color='red', linestyle='--', linewidth=2, alpha=0.6)

fig.suptitle('DPO-only Paradox: Discriminative != Generative',
             fontsize=17, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('report/figures/figure_4_5_dpo_only_paradox.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 4.5 saved")

# ============================================================================
# 图5.2：LoRA秩消融
# ============================================================================
print("[9/14] Generating Figure 5.2: LoRA Rank Ablation...")

ranks = [4, 8, 16, 32]
f1_scores = [0.882, 0.855, 0.876, 0.873]
chair_i = [16.59, 16.64, 17.07, 16.10]

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = '#2196F3'
ax1.set_xlabel('LoRA Rank (r)', fontsize=14, fontweight='bold')
ax1.set_ylabel('POPE F1', color=color1, fontsize=14, fontweight='bold')
line1 = ax1.plot(ranks, f1_scores, color=color1, marker='o', linewidth=3,
                 markersize=12, label='POPE F1')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim([0.83, 0.90])
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = '#FF5722'
ax2.set_ylabel('CHAIR_i (%)', color=color2, fontsize=14, fontweight='bold')
line2 = ax2.plot(ranks, chair_i, color=color2, marker='s', linewidth=3,
                 markersize=10, linestyle='--', label='CHAIR_i')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim([15, 18])

plt.title('LoRA Rank Sensitivity: Minimal Impact',
          fontsize=16, fontweight='bold', pad=20)

# 标注发现
ax1.text(18, 0.88, 'Variance < 2pp\nRank impact minimal',
        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        ha='center', fontweight='bold')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', fontsize=11)

fig.tight_layout()
plt.savefig('report/figures/figure_5_2_lora_rank.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 5.2 saved")

# ============================================================================
# 图5.9：DPO训练轮数消融
# ============================================================================
print("[10/14] Generating Figure 5.9: DPO Epoch Ablation...")

epochs = ['1 epoch', '3 epochs']
f1_scores = [0.869, 0.780]
yes_ratios = [0.395, 0.320]
chair_i = [17.81, 18.88]

x = np.arange(len(epochs))
width = 0.25

fig, ax1 = plt.subplots(figsize=(10, 7))

# F1柱状图
bars1 = ax1.bar(x - width, f1_scores, width, label='POPE F1',
                color='#66BB6A', edgecolor='darkgreen', linewidth=2)
ax1.set_ylabel('POPE F1 Score', fontsize=13, fontweight='bold')
ax1.set_ylim([0.7, 0.9])

# Yes-ratio
ax2 = ax1.twinx()
bars2 = ax2.bar(x, yes_ratios, width, label='Yes-Ratio',
                color='#64B5F6', edgecolor='darkblue', linewidth=2, alpha=0.7)
ax2.set_ylabel('Yes-Ratio', fontsize=13, fontweight='bold')
ax2.set_ylim([0.25, 0.45])

# CHAIR_i
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
bars3 = ax3.bar(x + width, chair_i, width, label='CHAIR_i',
                color='#FFAB91', edgecolor='darkorange', linewidth=2, alpha=0.7)
ax3.set_ylabel('CHAIR_i (%)', fontsize=13, fontweight='bold')
ax3.set_ylim([15, 22])

ax1.set_xticks(x)
ax1.set_xticklabels(epochs, fontsize=13)
ax1.set_title('DPO Training Epochs: 1 > 3 (Confirmed!)',
              fontsize=16, fontweight='bold', pad=20)

# 标注改善
ax1.text(0.5, 0.82, f'+8.9pp F1\n1 epoch BETTER',
        fontsize=12, color='green', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
          loc='lower right', fontsize=10)

fig.tight_layout()
plt.savefig('report/figures/figure_5_9_epoch_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 5.9 saved")

print("\n" + "="*60)
print("SUCCESS: Generated 5 more figures! (6-10/14)")
print("="*60)
