#!/usr/bin/env python3
"""批量生成报告图表"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('report/figures', exist_ok=True)

print("=" * 60)
print("Start generating figures...")
print("=" * 60)

# ============================================================================
# 图5.3：数据规模曲线（"少即是多"核心发现）
# ============================================================================
print("\n[1/14] Generating Figure 5.3: Data Scale Curve...")

data_scales = ['5K', '10K', '25K', '50K']
pope_f1 = [0.922, 0.903, 0.893, 0.855]
yes_ratio = [0.457, 0.446, 0.456, 0.521]
training_time_h = [0.5, 1.0, 2.5, 5.0]

fig, ax1 = plt.subplots(figsize=(12, 7))

# 主轴：F1分数
color1 = '#2196F3'
ax1.set_xlabel('SFT数据规模', fontsize=14, fontweight='bold')
ax1.set_ylabel('POPE F1分数', color=color1, fontsize=14, fontweight='bold')
line1 = ax1.plot(data_scales, pope_f1, color=color1, marker='o', linewidth=3,
                 markersize=12, label='POPE F1', zorder=3)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
ax1.set_ylim([0.84, 0.93])
ax1.grid(True, alpha=0.3, linestyle='--')

# 标注最优点
ax1.annotate('⭐ 最优: F1=0.922\n训练仅0.5小时',
             xy=(0, 0.922), xytext=(1.5, 0.915),
             arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
             fontsize=12, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff9c4',
                      edgecolor='red', linewidth=2, alpha=0.9))

# 次轴：Yes-ratio
ax2 = ax1.twinx()
color2 = '#FF5722'
ax2.set_ylabel('Yes-Ratio（偏差指标）', color=color2, fontsize=14, fontweight='bold')
line2 = ax2.plot(data_scales, yes_ratio, color=color2, marker='s', linewidth=3,
                 markersize=10, linestyle='--', label='Yes-Ratio', zorder=2)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
ax2.set_ylim([0.42, 0.54])
ax2.axhline(y=0.50, color='green', linestyle=':', linewidth=2, alpha=0.6, label='理想值0.50')

# 标注问题点
ax2.annotate('❌ 50K偏差过大\n+9.0pp偏移',
             xy=(3, 0.521), xytext=(2.2, 0.535),
             arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
             fontsize=11, color='darkred', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', alpha=0.8))

# 标题
plt.title('"少即是多"现象：SFT数据规模对幻觉的影响',
          fontsize=16, fontweight='bold', pad=20)

# 图例
lines = line1 + line2 + [plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2)]
labels = ['POPE F1分数', 'Yes-Ratio偏差', '理想Yes-Ratio']
ax1.legend(lines, labels, loc='lower left', fontsize=11, framealpha=0.9)

fig.tight_layout()
plt.savefig('report/figures/figure_5_3_data_scale_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 5.3 saved")

# ============================================================================
# 图5.6：DPO Beta敏感性曲线
# ============================================================================
print("[2/14] Generating Figure 5.6: DPO Beta Sensitivity...")

betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
f1_scores = [0.000, 0.051, 0.780, 0.828, 0.841, 0.846]
yes_ratios = [0.000, 0.020, 0.320, 0.359, 0.370, 0.374]

fig, ax1 = plt.subplots(figsize=(12, 7))

# F1曲线
color1 = '#4CAF50'
ax1.set_xlabel('DPO Beta参数（β）', fontsize=14, fontweight='bold')
ax1.set_ylabel('POPE F1分数', color=color1, fontsize=14, fontweight='bold')
ax1.plot(betas, f1_scores, color=color1, marker='o', linewidth=3,
         markersize=12, label='F1分数')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
ax1.set_ylim([-0.05, 0.90])
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# 崩溃区域阴影
ax1.axvspan(0.01, 0.09, alpha=0.2, color='red', label='崩溃区域（β<0.1）')
ax1.text(0.03, 0.4, '⚠️ 模型崩溃\nF1<0.1', fontsize=12, color='darkred',
         fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.8))

# 稳定区域
ax1.axvspan(0.1, 1.0, alpha=0.1, color='green')
ax1.text(0.4, 0.75, '✓ 稳定区域\nβ≥0.1', fontsize=11, color='darkgreen',
         fontweight='bold', ha='center')

# 最优点标注
ax1.annotate('⭐ 最优β=1.0\nF1=0.846',
             xy=(1.0, 0.846), xytext=(0.5, 0.82),
             arrowprops=dict(arrowstyle='->', color='gold', lw=2.5),
             fontsize=12, color='#F57C00', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff9c4',
                      edgecolor='orange', linewidth=2))

# Yes-ratio次轴
ax2 = ax1.twinx()
color2 = '#2196F3'
ax2.set_ylabel('Yes-Ratio', color=color2, fontsize=14, fontweight='bold')
ax2.plot(betas, yes_ratios, color=color2, marker='s', linewidth=2.5,
         markersize=10, linestyle='--', alpha=0.7, label='Yes-Ratio')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
ax2.set_ylim([0, 0.45])

plt.title('DPO Beta超参数敏感性分析', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=11)

fig.tight_layout()
plt.savefig('report/figures/figure_5_6_beta_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 5.6 saved")

# ============================================================================
# 图4.1：POPE三模型跨分割对比
# ============================================================================
print("[3/14] Generating Figure 4.1: POPE Three Models...")

splits = ['Random\n随机', 'Popular\n流行', 'Adversarial\n对抗']
base_f1 = [0.879, 0.876, 0.882]
sft50k_f1 = [0.855, 0.852, 0.858]
true_opt_f1 = [0.889, 0.887, 0.891]

x = np.arange(len(splits))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width, base_f1, width, label='Base模型',
               color='#90CAF9', edgecolor='#1976D2', linewidth=2)
bars2 = ax.bar(x, sft50k_f1, width, label='SFT 50K',
               color='#FFAB91', edgecolor='#D84315', linewidth=2)
bars3 = ax.bar(x + width, true_opt_f1, width, label='True Optimal',
               color='#A5D6A7', edgecolor='#388E3C', linewidth=2)

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('POPE分割类型', fontsize=14, fontweight='bold')
ax.set_ylabel('F1分数', fontsize=14, fontweight='bold')
ax.set_title('POPE三模型跨分割性能对比', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=12)
ax.legend(fontsize=12, loc='lower right')
ax.set_ylim([0.83, 0.92])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加关键洞察标注
ax.text(2, 0.875, '对抗分割最难\n所有模型退化', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
        ha='center')

plt.tight_layout()
plt.savefig('report/figures/figure_4_1_pope_three_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 4.1 saved")

# ============================================================================
# 图4.2：Yes-Ratio轨迹
# ============================================================================
print("[4/14] Generating Figure 4.2: Yes-Ratio Trajectory...")

models = ['Base', 'SFT 5K', 'SFT 50K', 'True\nOptimal']
yes_ratios = [0.431, 0.457, 0.521, 0.413]
colors = ['#78909C', '#66BB6A', '#EF5350', '#42A5F5']

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(models, yes_ratios, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

# 理想线
ax.axhline(y=0.50, color='green', linestyle='--', linewidth=2, label='理想值0.50')
ax.axhline(y=0.43, color='gray', linestyle=':', linewidth=2, alpha=0.6, label='Base基线0.431')

# 标注
for i, (bar, ratio) in enumerate(zip(bars, yes_ratios)):
    height = bar.get_height()
    delta = ratio - 0.431
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{ratio:.3f}\n({delta:+.1%})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 特殊标注
    if i == 2:  # SFT 50K
        ax.annotate('❌ Yes-bias问题\n+9.0pp偏移',
                   xy=(i, ratio), xytext=(i-0.3, 0.55),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.8))
    elif i == 3:  # True Optimal
        ax.annotate('✓ DPO修正\n最接近理想',
                   xy=(i, ratio), xytext=(i+0.2, 0.45),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=11, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

ax.set_ylabel('Yes-Ratio（肯定回答比例）', fontsize=14, fontweight='bold')
ax.set_title('Yes-Bias演变轨迹：从Base到True Optimal', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([0.35, 0.58])
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/figures/figure_4_2_yes_ratio_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 4.2 saved")

# ============================================================================
# 图4.3：CHAIR指标对比
# ============================================================================
print("[5/14] Generating Figure 4.3: CHAIR Comparison...")

models = ['Base', 'SFT 50K', 'True Optimal']
chair_i = [33.31, 16.64, 20.12]
recall = [81.37, 64.89, 74.24]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 7))

# CHAIR_i柱状图（越低越好）
color1 = '#E57373'
bars1 = ax1.bar(x - width/2, chair_i, width, label='CHAIR_i (幻觉率%)',
                color=color1, edgecolor='darkred', linewidth=2, alpha=0.8)
ax1.set_ylabel('CHAIR_i 幻觉率 (%)', fontsize=14, fontweight='bold', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim([0, 40])

# 召回率柱状图（越高越好）
ax2 = ax1.twinx()
color2 = '#81C784'
bars2 = ax2.bar(x + width/2, recall, width, label='召回率 (%)',
                color=color2, edgecolor='darkgreen', linewidth=2, alpha=0.8)
ax2.set_ylabel('物体召回率 (%)', fontsize=14, fontweight='bold', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim([50, 90])

# 数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_xlabel('模型', fontsize=14, fontweight='bold')
ax1.set_title('CHAIR生成式幻觉 vs 物体召回率权衡', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

# 标注最优
ax1.text(2, 25, '⭐ True Optimal\n最佳平衡点', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#fff9c4', edgecolor='orange', linewidth=2),
        ha='center', fontweight='bold')

fig.tight_layout()
plt.savefig('report/figures/figure_4_3_chair_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK - Figure 4.3 saved")

print("\n" + "="*60)
print("SUCCESS: Generated 5 key figures!")
print("="*60)
