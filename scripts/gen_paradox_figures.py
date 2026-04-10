"""生成 DPO-only 悖论的三张独立图片，用于 PPT 展示"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

out_dir = r'D:\硕士阶段\研一下\大模型后训练\Course design\vqa-hallucination\report\figures'

models = ['Base', 'SFT 50K', 'DPO-only', 'True\nOptimal']
colors = ['#60a5fa', '#f87171', '#4ade80', '#fbbf24']

# ====== 图1: 判别能力 (POPE F1) ======
pope_f1 = [0.879, 0.853, 0.900, 0.889]
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(models, pope_f1, color=colors, edgecolor='#334155', linewidth=1.2, width=0.6)
ax.set_ylabel('POPE F1 Score', fontsize=14, fontweight='bold')
ax.set_title('判别能力 (Discriminative)', fontsize=16, fontweight='bold', pad=12)
ax.set_ylim(0.82, 0.95)
for bar, v in zip(bars, pope_f1):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f'{v:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# BEST 标注在数值上方
ax.annotate('BEST!', xy=(2, 0.918), fontsize=13, fontweight='bold',
            color='#dc2626', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef08a', edgecolor='#f59e0b'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'paradox_1_pope.png'), dpi=200, bbox_inches='tight')
plt.close()

# ====== 图2: 生成质量 (CHAIR_i, 越低越好) ======
chair_i = [33.31, 16.64, 31.83, 20.12]
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(models, chair_i, color=colors, edgecolor='#334155', linewidth=1.2, width=0.6)
ax.set_ylabel('CHAIR_i (%)  ↓ 越低越好', fontsize=14, fontweight='bold')
ax.set_title('生成质量 (Generative)', fontsize=16, fontweight='bold', pad=12)
ax.set_ylim(0, 46)
for bar, v in zip(bars, chair_i):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# WORST 标注在数值上方
ax.annotate('WORST!', xy=(2, 37), fontsize=13, fontweight='bold',
            color='white', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dc2626', edgecolor='#b91c1c'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'paradox_2_chair.png'), dpi=200, bbox_inches='tight')
plt.close()

# ====== 图3: 能力保持 (MME) ======
mme = [2008.0, 1944.0, 1963.0, 1990.5]
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(models, mme, color=colors, edgecolor='#334155', linewidth=1.2, width=0.6)
ax.axhline(y=2008.0, color='#ef4444', linestyle='--', linewidth=2, label='Base 基线')
ax.set_ylabel('MME Total Score', fontsize=14, fontweight='bold')
ax.set_title('能力保持 (Preservation)', fontsize=16, fontweight='bold', pad=12)
ax.set_ylim(1900, 2060)
for bar, v in zip(bars, mme):
    ax.text(bar.get_x() + bar.get_width()/2, v + 6, f'{v:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'paradox_3_mme.png'), dpi=200, bbox_inches='tight')
plt.close()

print("Done: paradox_1_pope.png, paradox_2_chair.png, paradox_3_mme.png")
