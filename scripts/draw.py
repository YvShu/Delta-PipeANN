import re
import os
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ======================== 配置 ========================
# 设置全局绘图风格参数
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('xtick', labelsize=15)  # X轴刻度字体大小
plt.rc('ytick', labelsize=15)  # Y轴刻度字体大小

# 实验结果输出路径
base_path = r"/home/guyue/Delta-PipeANN/output/"
save_path = r"/mnt/hgfs/DataSet/SSDANNResults/"

# 可变参数
variable_params = {
    'dataset_size': ['1M'],
    'query_count': [16],
    'degree': [48],

    'memgraph_enabled': [True],
    # 'memgraph_sample': ['0.01'],
    # 'memgraph_L': [10],
    'memgraph_sample': ['0.01'],
    'memgraph_L': [10],

    'cache_enabled': [False],
    'cache_strategy': ['fifo'],
    # 'cache_strategy': ['fifo', 'lru', 'entrypage'],
    'cache_capacity': [1024],
    # 'cache_capacity': [1024, 2048, 3072, 4096],
    'cache_hop': [2],
    # 'cache_hop': [2, 8, 16],

    'relayout_enabled': [True],
    'relayout_strategy': ['default', '1bamg', '2bamg', '3bamg', '4bamg'],

    'search_strategy': ['pagese'],
}

# 要绘制的 X, Y 列
FIG_SIZE = (10, 8)                      # 画布大小
X_COLUMN = "Mean Lat"                   # X轴列名
Y_COLUMN = "Recall@10"                  # Y轴列名
TITLE = f"{Y_COLUMN} vs {X_COLUMN}"     # 图表标题
X_LABEL = X_COLUMN                      # X轴标签
Y_LABEL = Y_COLUMN                      # Y轴标签

MARKER_BY = ['degree']

save_path += X_COLUMN + " vs " + Y_COLUMN
for key in MARKER_BY:
    save_path = save_path + key + "_"
save_path += ".svg"

# 预定义颜色和标记（自动循环使用）
color_list = [
    "#E57373", "#64B5F6", "#81C784", "#FFB74D", "#9575CD", "#4DB6AC",
    "#F06292", "#7986CB", "#4FC3F7", "#AED581", "#BA68C8", "#FF8A65",
    "#4DD0E1", "#A1887F", "#90A4AE", "#404040",
]
marker_list = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

# ======================== 数据提取 ========================
COLUMN_NAMES = [
    'Ls', 'QPS', 'Mean Lat', '50 Lat', '90 Lat', '95 Lat', '99 Lat', '99.9 Lat',
    'Recall@10', 'Disk IOs', 'IO Ready', 'Last CPU', 'IO Wait', 'Cur CPU', 'Loop', 'Total'
]


def get_marker_key(cfg, marker_keys):
    """
    根据指定的 marker_keys 提取特征，生成唯一的 marker 标识。
    """
    if not marker_keys:
        return "default_marker"

    parts = []
    for k in marker_keys:
        # 提取指定键的值，如果配置中没有该键，则设为 N/A
        val = cfg.get(k, 'N/A')
        parts.append(f"{k}:{val}")
    return "|".join(parts)


def get_color_key(cfg, marker_keys):
    """
    将除 marker_keys 之外的所有参数拼接，生成唯一的 color 标识。
    """
    parts = []
    # 遍历所有键，排除用于 marker 的键
    # 使用 sorted 保证即使字典键顺序不同，生成的字符串也一致
    for k in sorted(cfg.keys()):
        if k not in marker_keys:
            parts.append(f"{k}:{cfg[k]}")

    if not parts:
        return "default_color"
    return "|".join(parts)

def parse_log_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or 'Ls' in line or '===' in line or 'QPS' in line:
                continue
            parts = line.split()
            if len(parts) == len(COLUMN_NAMES):
                try:
                    float(parts[0])
                    data.append([float(x) for x in parts])
                except:
                    continue
    if not data:
        raise ValueError(f"No valid data in {file_path}")
    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    return df.sort_values('Ls').reset_index(drop=True)

def generate_filename(cfg):
    parts = []

    # 数据规模
    dataset_size = cfg.get('dataset_size', '1M')
    parts.append(f"{dataset_size}")

    # 查询规模
    query_count = cfg.get('query_count', '16')
    parts.append(f"Q{query_count}")

    # 图的度
    degree = cfg.get('degree', '32')
    parts.append(f"R{degree}")

    # 内存图优化
    if cfg.get('memgraph_enabled', False):
        sample = cfg.get('memgraph_sample', 0.01)
        l_val = cfg.get('memgraph_L', 10)
        parts.append(f"+memgraph{sample}L{l_val}")
    else:
        parts.append("-memgraph")

    # 缓存优化
    if cfg.get('cache_enabled', False):
        cache_strategy = cfg.get('cache_strategy', 'fifo')
        if cache_strategy == 'fifo' or cache_strategy == 'lru':
            cache_capacity = cfg.get('cache_capacity', 1024)
            parts.append(f"+cache{cache_capacity}{cache_strategy}")
        elif cache_strategy == 'entrypage':
            cache_hop = cfg.get('cache_hop', 2)
            parts.append(f"+cache{cache_hop}{cache_strategy}")
    else:
        parts.append("-cache")

    # SSD布局优化
    if cfg.get('relayout_enabled', False):
        relayout_strategy = cfg.get('relayout_strategy', 'default')
        if relayout_strategy == 'default':
            parts.append(f"+relayout")
        else:
            parts.append(f"+relayout{relayout_strategy}")
    else:
        parts.append("-relayout")

    # 搜索方式
    search_strategy = cfg.get('search_strategy', 'pagese')
    parts.append(f"+{search_strategy}")

    return "_".join(parts) + ".txt"

def generate_group_config(cfgs):
    var_keys = list(cfgs.keys())
    var_vals = [cfgs[k] for k in var_keys]

    combos = list(product(*var_vals))

    configs = []
    for combo in combos:
        cfg = variable_params.copy()
        for k, v in zip(var_keys, combo):
            cfg[k] = v
        configs.append(cfg)

    print(configs)
    return configs

# ======================== 主程序 ========================
def main():
    # 生成所需的配置
    configs = generate_group_config(variable_params)

    combo_data = {}
    marker_map = {}
    color_map = {}
    for cfg in configs:
        # 生成文件名
        filename = generate_filename(cfg)
        file_path = os.path.join(base_path, filename)

        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        # 解析
        try:
            df = parse_log_file(file_path)
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")
            continue

        if X_COLUMN not in df.columns or Y_COLUMN not in df.columns:
            print(f"Columns {X_COLUMN} or {Y_COLUMN} not found in {file_path}")
            continue

        # 生成图例标签
        label_parts = []
        memgraph_enabled = False
        cache_enabled = False
        relayout_enabled = False
        cache_capacity = 0
        cache_hop = 0
        cache_strategy = ''
        relayout_enabled = False
        for k in variable_params.keys():
            v = cfg.get(k)

            if k == 'query_count':
                label_parts.append(f"Q{v}")
            elif k == 'degree':
                label_parts.append(f"R{v}")
            elif k == 'memgraph_enabled':
                if v:
                    memgraph_enabled = True
                    label_parts.append(f"+mem")
                else:
                    label_parts.append(f"-mem")
            elif k == 'memgraph_sample':
                if memgraph_enabled:
                    label_parts.append(f"{v}")
            elif k == 'memgraph_L':
                if memgraph_enabled:
                    label_parts.append(f"L{v}")
            elif k == 'cache_enabled':
                if v:
                    cache_enabled = True
                    label_parts.append(f"+cache")
                else:
                    label_parts.append(f"-cache")
            elif k == 'cache_strategy':
                cache_strategy = v
            elif k == 'cache_capacity':
                cache_capacity = int(v)
                if cache_enabled:
                    if cache_strategy == 'fifo' or cache_strategy == 'lru':
                        label_parts.append(f"{cache_capacity}{cache_strategy}")
            elif k == 'cache_hop':
                cache_hop = int(v)
                if cache_enabled:
                    if cache_strategy == 'entrypage':
                        label_parts.append(f"{cache_hop}{cache_strategy}")

            elif k == 'relayout_enabled':
                if v:
                    relayout_enabled = True
                    label_parts.append(f"+relayout")
                else:
                    label_parts.append(f"-relayout")
            elif k == 'relayout_strategy':
                if relayout_enabled:
                    if v == 'default':
                        pass
                    else:
                        label_parts.append(f"{v}")
                else:
                    pass
            elif k == 'search_strategy':
                label_parts.append(f"{v}")

        label = " ".join(label_parts)
        print(label)
        # ==================== 通用颜色与形状分配逻辑 ====================
        # 传入全局配置 MARKER_BY
        m_key = get_marker_key(cfg, MARKER_BY)
        c_key = get_color_key(cfg, MARKER_BY)

        # 动态分配 Marker 和 Color
        if m_key not in marker_map:
            marker_map[m_key] = marker_list[len(marker_map) % len(marker_list)]
        if c_key not in color_map:
            color_map[c_key] = color_list[len(color_map) % len(color_list)]

        combo_data[label] = {
            'x': df[X_COLUMN].values,
            'y': df[Y_COLUMN].values,
            'filename': filename,
            'cfg': cfg,
            'marker': marker_map[m_key],
            'color': color_map[c_key]
        }

    fig, ax = plt.subplots(figsize = FIG_SIZE)
    for idx, (label, data) in enumerate(combo_data.items()):
        # color = color_list[idx % len(color_list)]
        # marker = marker_list[(idx // len(color_list)) % len(marker_list)]

        ax.plot(data['x'], data['y'],
                color=data['color'],
                marker=data['marker'],
                markersize=6.5,
                markeredgewidth=1.0, markerfacecolor='none',
                markeredgecolor=data['color'], linewidth=1.5, label=label)

    ax.set_xlabel(X_LABEL, fontsize=15)
    ax.set_ylabel(Y_LABEL, fontsize=15)
    ax.set_title(TITLE, fontsize=17, fontweight='bold', pad=5)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='best', fontsize=12, frameon=True, framealpha=0.9)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图像已保存至: {save_path}")

if __name__ == "__main__":
    main()



