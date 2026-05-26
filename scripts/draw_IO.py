import re
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.scale import FuncScale
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl


# 设置全局绘图风格参数
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('xtick', labelsize=15)  # X轴刻度字体大小
plt.rc('ytick', labelsize=15)  # Y轴刻度字体大小

dataset = "sift-2M"
workload = "workload2"
base_path = fr"/home/guyue/Delta-PipeANN/output/"
save_name = dataset + ""
k = 100
q = 4
p = 3000

colors = [
    "#E57373",  # 柔和红色
    "#64B5F6",  # 淡蓝色
    "#81C784",  # 淡绿色
    "#FFB74D",  # 柔和橙色
    "#9575CD",  # 淡紫色
    "#4DB6AC",  # 青绿色
    "#F06292",  # 粉色
    "#7986CB",  # 淡靛蓝
    "#4FC3F7",  # 天蓝色
    "#AED581",  # 浅绿色
    "#BA68C8",  # 淡紫红色
    "#FF8A65",  # 淡珊瑚色
    "#4DD0E1",  # 青色
    "#A1887F",  # 棕色
    "#90A4AE",  # 蓝灰色
    "#404040",
]

# 设置配色方案
group_config = {
    f"Vamana32_lvq_node_search": {'marker': '^', 'label': "Node Search"},
    f"Vamana32_lvq_page_search": {'marker': 'o', 'label': "Page Search"},
    f"Vamana32_lvq+_node_search": {'marker': 'o', 'label': "Node Search*"},
}

# 设置画布大小
fig, ax = plt.subplots(figsize=(10, 6))

idx = 0
step = 0
for _, config in enumerate(group_config):
    file_path = os.path.join(base_path, config + '.txt')
    with open(file_path, 'r') as f:
        content = f.read()

    # 使用正则表达式提取所有步骤数和对应值
    pattern =  r'search current time:\s*(\d+).*?\n\s*50\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE)

    values_x = []
    values_y = []
    step = 1
    for match in matches:
        # step = int(match[0])
        time_val = float(match[1])
        values_x.append(step)
        values_y.append(time_val)
        step += 1

    ax.plot(
        values_x,
        values_y,
        color=colors[idx],
        # marker=group_config[config]['marker'],
        markersize = 6.5,
        markeredgewidth = 1.0,
        markerfacecolor = "none",
        markeredgecolor = colors[idx],
        label = group_config[config]['label'],
        linewidth = 1.5,
    )
    idx += 1

# 设置坐标轴标签和标题
ax.set_xlabel("step", fontsize=15)
ax.set_ylabel("Disk IOs", fontsize=15)
plt.title(f'Disk IOs (L: 50)', fontsize=17, fontweight='bold', pad=5)

n_values = np.arange(0, step + 1)
n = len(n_values)
tick_positions = np.arange(0, n, 20)
tick_labels = np.arange(0, n, 20).astype(str)
plt.xticks(ticks=tick_positions, labels=tick_labels)


# 添加网格
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
# ax.legend(loc='upper left', fontsize=15, frameon=True, framealpha=0.9)
ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9, bbox_to_anchor=(1.2, 0.0))

# 设置x轴刻度为整数（因为step是整数）
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# 优化布局
plt.tight_layout(pad=2.0)  # 增加padding为图例留出空间

# 保存为高质量科研图像
plt.savefig(f'/mnt/hgfs/DataSet/experiment_results/DiskIOs-L50.svg', dpi=150, bbox_inches='tight')
plt.show()
















