# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: main.py
@Project: common
@Time: 2024/11/19   01:58
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
@Brief: 与评价方法相关的都放在这个模块中。
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import tensorflow as tf
from shap.plots import colors
import matplotlib
from shap. plots._force_matplotlib import format_data, draw_bars, draw_labels
from shap.utils import approximate_interactions, convert_name,_general
from matplotlib import lines
import matplotlib.patches as patches
from tensorflow.keras import backend


# @saving.register_keras_serializable(package="Custom")
def r2(y_true, y_pred):
    """
    r2还可以用mse来计算.
    y_mean = np.mean(y_true)
    r2 = 1 - (mse * len(y_true)) / np.sum((y_true - y_mean)**2)
    :param y_true:
    :param y_pred:
    :param sample_weight:
    :return:
    """
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # 加入 epsilon 防止除零错误
    return 1 - ss_res / (ss_tot + backend.epsilon())


def _entropy(matrix):
    """
    定义熵权法函数，返回综合得分。
    :param matrix: matrix为需要进行熵权法的pandas
    :return:返回估计出来的权重矩阵
    """
    matrix_ = matrix.copy(deep=True).replace(0, 0.0001)
    for name, item in matrix.items():
        max_ = item.max()
        min_ = item.min()
        for index, row in item.items():
            matrix_.loc[index, name] = (max_ - matrix_.loc[index, name]) / (max_ - min_)
    project, indicator = matrix_.shape
    p = pd.DataFrame(np.zeros([project, indicator], dtype='float64'), columns=matrix_.columns)
    for i in range(indicator):
        p.iloc[:, i] = matrix_.iloc[:, i] / matrix_.iloc[:, i].sum()
    E = -1 / np.log(project) * (p * np.log(p)).sum()
    return (1 - E) / sum((1 - E))


def entropy(matrix):
    _matrix = matrix.copy(deep=True).replace(0, 0.001)
    _matrix = (_matrix - _matrix.min()) / (_matrix.max() - _matrix.min())
    p, i = _matrix.shape
    _matrix = _matrix / _matrix.sum()
    _matrix.replace(0, 0.0001, inplace=True)
    E = -1 / np.log(p) * (_matrix * np.log(_matrix)).sum()
    return (1 - E) / sum((1 - E))


def topsis(matrix, W):
    CCs = []
    m, n = matrix.shape
    pis = np.square(np.ones((m, n)) - matrix)
    nis = np.square(matrix)
    for i in range(m):
        posd = np.sqrt((pis.iloc[i, :] * W).sum())
        negd = np.sqrt((nis.iloc[i, :] * W).sum())
        CCs.append(negd / (posd + negd))
    return pd.Series(CCs, index=matrix.index)


def var_contribution(data):
    """
    该方法用于输出特征值、方差贡献率以及累计方差贡献率
    :param data:未标准化的矩阵
    :return:
    """
    labels = data.columns
    X_scale = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    # 按列求解相关系数矩阵，存入cor中(rowvar=0指定按列求解相关系数矩阵)
    cor = np.corrcoef(X_scale, rowvar=False)
    # 求解相关系数矩阵特征值与特征向量，并按照特征值由大到小排序
    # 注意numpy中求出的特征向量是按列排列而非行，因此注意将矩阵转置
    eigvalue, eigvector = np.linalg.eig(cor)
    eigdf = pd.DataFrame(eigvector.T).join(pd.DataFrame(eigvalue, dtype=float, columns=["Eigvalue"]))
    eigdf.index = labels
    # 将特征向量按特征值由大到小排序
    eigdf = eigdf.sort_values("Eigvalue")[::-1]
    eig_df = eigdf[["Eigvalue"]]
    # 计算每个特征值的方差贡献率，存入varcontribution中
    eig_df["Proportion"] = eig_df.apply(lambda x: x / eig_df["Eigvalue"].sum())
    # 累积方差贡献率
    eig_df["Cumulative"] = eig_df["Proportion"].cumsum()
    # 将特征值、方差贡献率，累积方差贡献率写入DataFrame
    return eig_df


def best_worst_fij(matrix, vec):
    """
    准则函数:确定最优值与最劣值
    a ：指标值矩阵
    b ：表示指标是极大型还是极小型的数组
    """
    f = np.zeros((vec.shape[0], 2))  # 初始化最优值矩阵
    for i in range(vec.shape[0]):
        if vec[i] == 'max':
            f[i, 0] = matrix.max(axis=0)[i]  # 最优值
            f[i, 1] = matrix.min(axis=0)[i]  # 最劣值
        elif vec[i] == 'min':
            f[i, 0] = matrix.min(axis=0)[i]  # 最优值
            f[i, 1] = matrix.max(axis=0)[i]  # 最劣值
    return f


def vikor(matrix, vec, w, v=0.5):
    """
    VIKOR
    metrix ：指标值矩阵
    vec ：最优值和最劣值矩阵
    w ：指标权重
    y ：是否绘图
    """

    def SR(matrix, f, w):
        """
        计算效用值S_i 和遗憾值 R_i
        a ：指标值矩阵
        b ：最优值和最劣值矩阵
        c ：指标权重
        """
        s = np.zeros(matrix.shape[0])
        r = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            k = 0
            o = 0
            for j in range(matrix.shape[1]):
                k = k + w[j] * abs((f[j, 0] - matrix.iloc[i, j]) / (f[j, 0] - f[j, 1]))  # 与最优值总的相对距离
                u = w[j] * abs((f[j, 0] - matrix.iloc[i, j]) / (f[j, 0] - f[j, 1]))  # 与给定指标最优值的相对距离
                if u > o:
                    o = u  # 求最大遗憾值
                    r[i] = o
            s[i] = k
        return s, r

    def Q(s, r, v):
        """
        计算折中值 Q_i
        s ：效益值数组
        r ：遗憾值数组
        w ：折衷系数
        """
        q = np.zeros(s.shape[0])
        for i in range(s.shape[0]):
            # q[i] = v *(s[i] - min(s)) / (max(s) - min(s)) +(1 - v)*(r[i] - min(r)) / (max(r) - min(r))
            q[i] = v * (s[i] - min(s)) / (max(s) - min(s)) + (1 - v) * (r[i] - min(r)) / (max(r) - min(r))
        return q

    s, r = SR(matrix, best_worst_fij(matrix, vec), w)
    q = Q(s, r, v)
    return s, r, q


def gra(matrix, w):
    # 熵权-灰色关联
    A1 = matrix.max()
    data = (matrix - A1).abs()
    max_ = data.max().max()
    min_ = data.min().min()
    df_r = (min_ * w + data) / (max_ * w + data)
    return df_r.mean(axis=1)


def shap_summary_plot(
        shap_values,
        features,
        feature_names=None,
        title="SHAP值 模型特征贡献图",
        figsize=(8,6),
        max_display=20,
        savepath=None,
        alpha=1
):
    idx2cat = None
    # plt.rcParams['font.family'] = ['Times New Roman', 'Heiti TC', 'Heiti TC', ]
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    axis_color = "#333333"
    color = colors.blue_rgb
    cmap = colors.red_blue
    row_height = 0.4
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    feature_inds = feature_order[:max_display]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax2 = ax.twiny()
    y_pos = np.arange(len(feature_inds))
    global_shap_values = np.abs(shap_values).mean(0)
    ax2.barh(y_pos, global_shap_values[feature_inds], 0.75, align='center', color=color, alpha=0.25,
             label="mean(|SHAP value|)", zorder=-3)

    ax.axvline(x=0, color="r", zorder=-2, linewidth=2, linestyle='--')
    for pos, i in enumerate(feature_order):
        # ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]
        values = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]:  # check categorical feature
                colored_feature = False
            else:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        except Exception:
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            if vmin > vmax:  # fixes rare numerical precision issues
                vmin = vmax
            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"
            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            ax.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777",
                       s=16, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                       cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                       c=cvals, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
        else:
            ax.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                       color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
    # SHAP value
    ax.set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=11)
    ax2.set_xticks([])
    ax.tick_params('x', labelsize=12)
    ax.set_ylim(-1, len(feature_order))

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])

    cb = plt.colorbar(m, ax=plt.gca(), ticks=[0, 1], aspect=80)
    cb.set_ticklabels(["低", "高"])
    cb.set_label("特征值", size=13, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)

    for row in ("right", "top", "left"):
        ax.spines[row].set_visible(False)
        ax2.spines[row].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax2.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('k')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.grid(False)

    # plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()


def dependency_plot(ind, shap_values, features, feature_names, ax=None, alpha=1, xmin=None, xmax=None, ymin=None, ymax=None):
    x_jitter = 0
    interaction_index = "auto"
    categorical_interaction = False
    ind = convert_name(ind, shap_values, feature_names)
    if interaction_index == "auto":
        interaction_index = approximate_interactions(ind, shap_values, features)[0]
    interaction_index = convert_name(interaction_index, shap_values, feature_names)
    cmap = colors.red_blue
    dot_size = 16
    axis_color = "#333333"
    oinds = np.arange(shap_values.shape[0])
    xv = _general.encode_array_if_needed(features[oinds, ind])
    xd = features[oinds, ind]
    s = shap_values[oinds, ind]
    if isinstance(xd[0], str):
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    name = feature_names[ind]
    color_norm = None
    interaction_feature_values = _general.encode_array_if_needed(features[:, interaction_index])
    cv = interaction_feature_values
    cd = features[:, interaction_index]
    clow = np.nanpercentile(cv.astype(float), 5)
    chigh = np.nanpercentile(cv.astype(float), 95)
    if clow == chigh:
        clow = np.nanmin(cv.astype(float))
        chigh = np.nanmax(cv.astype(float))
    if isinstance(cd[0], str):
        cname_map = {}
        for i in range(len(cv)):
            cname_map[cd[i]] = cv[i]
        cnames = list(cname_map.keys())
        categorical_interaction = True
    elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
        categorical_interaction = True

    if categorical_interaction and clow != chigh:
        clow = np.nanmin(cv.astype(float))
        chigh = np.nanmax(cv.astype(float))
        bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N - 1))
        color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N - 1)
    if x_jitter > 0:
        if x_jitter > 1:
            x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)  # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size=len(xv)) * jitter_amount) - (jitter_amount / 2)
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)

    # plot the nan values in the interaction feature as grey
    cvals = interaction_feature_values[oinds].astype(np.float64)
    cvals_imp = cvals.copy()
    cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
    cvals[cvals_imp > chigh] = chigh
    cvals[cvals_imp < clow] = clow
    p = ax.scatter(
        xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
        cmap=cmap, alpha=alpha,
        norm=color_norm, rasterized=len(xv) > 500
    )
    p.set_array(cvals[xv_notnan])
    if xmin is not None or xmax is not None:
        if isinstance(xmin, str) and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if isinstance(xmax, str) and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))
        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv)) / 20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin) / 20
        ax.set_xlim(xmin, xmax)

    xlim = ax.get_xlim()
    p = ax.scatter(
        xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
        linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
        vmin=clow, vmax=chigh
    )
    p.set_array(cvals[xv_nan])

    ax.set_xlim(xlim)
    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=8)
    ax.set_ylabel("SHAP value for\n%s" % name, color=axis_color, fontsize=8)
    if (ymin is not None) or (ymax is not None):
        if ymin is None:
            ymin = -ymax
        if ymax is None:
            ymax = -ymin
        ax.set_ylim(ymin, ymax)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if isinstance(xd[0], str):
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, fontdict=dict(rotation='vertical', fontsize=8))



def feature_importance_plot(weight, save_path=None):
    weight['累积重要性'] = np.cumsum(weight['特征重要性']) / np.sum(weight['特征重要性'])
    # 计算累积特征重要性
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=300)
    data = weight.reset_index().iloc[:20][::-1]
    ax2 = ax.twiny()
    ax2.barh(data["index"], data['特征重要性'], height=0.7, align='center', color='b', label='特征重要性')
    # ax.bar_label(aa, label_type='center', padding=1, color='w')
    for i, y in enumerate(data['特征重要性']):
        ax.text(y + 0.3, i, f"{y:.2f}", ha='left', va='center', fontsize=10, fontweight='heavy', color='k',
                bbox=dict(pad=0, facecolor='none', edgecolor='none'))

    ax2.plot(data['累积重要性'] * 30, data['index'], color='r', marker='o', label='累积重要性')
    # # 标注累积特征重要性关键点
    for i, value in enumerate(data['累积重要性'][::-1]):
        ax.text(value * 30, data["index"][::-1][i], f'{value:.1%}', fontsize=9)
    ax.axvline(15, color='grey', linewidth=1, linestyle='--', label='50% 特征贡献线')

    # 隐藏负值区域的x轴刻度
    ax.set_xticks([i for i in range(0, 31, 5) if i >= 0])
    # ax2.set_xticklabels([f"{i/10:.0%}" for i in range(11)])

    # ax.set_xticks([tick for tick in ax.get_xticks() if tick >= 0])
    ax.tick_params(axis='y', labelsize=13)  # 仅调整 y 轴字体
    ax.set_xlim(-8, max(weight['特征重要性']) + 3)
    ax2.set_xlim(-8, max(weight['特征重要性']) + 3)
    # 美化图表
    ax.spines['left'].set_position(('data', 0))
    ax2.spines['left'].set_position(('data', 0))
    ax.spines['left'].set_color('k')
    ax.spines['top'].set_color('k')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('k')
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('k')
    ax2.spines['bottom'].set_linewidth(0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.tick_top()  # x轴刻度移至上方
    ax.xaxis.set_label_position('top')  # x轴标签移至上方
    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_label_position('bottom')
    # 自定义刻度位置
    ax2.set_xticks([i * 3 for i in range(11)])
    ax2.set_xticklabels([f"{i / 10:.0%}" for i in range(11)])
    ax.text(1, ax.get_ylim()[1] * 1.1,
            'CatBoost 模型特征重要性排序图',
            fontsize=15,
            fontweight='bold')
    # ax.set_title('基于 CatBoost 的特征重要性排序图', loc='left', fontsize=15, fontweight='bold')
    # 调整网格样式
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.grid(False)
    plt.legend(['累积重要性', '特征重要性'], ncols=3, loc='lower center', frameon=False)
    plt.subplots_adjust(left=0.1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)
    plt.show()


def ensure_not_numpy(x):
    """

    :param x:
    :return:
    """
    if isinstance(x, bytes):
        return x.decode()
    elif isinstance(x, np.str_):
        return str(x)
    elif isinstance(x, np.generic):
        return float(x.item())
    else:
        return x


def draw_additive_plot(base_values, shap_value, ax, fig, feature_names=None, save_path=None):
    """
    绘制 SHAP 贡献图，展示各个特征的 SHAP 贡献度。

    参数:
    - base_values: float，模型的基准预测值
    - shap_value: np.array，SHAP 贡献值数组
    - ax: matplotlib.axes.Axes，绘图的子图对象
    - fig: matplotlib.figure.Figure，绘图的 Figure 对象
    - feature_names: list，可选，特征名称列表

    返回:
    - 无返回值，直接在 ax 上绘制图形
    """
    # 确保 shap_value 是二维数组
    shap_values = np.reshape(shap_value, (1, len(shap_value)))

    # 处理特征名称
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]
    # 仅保留非零 SHAP 贡献值的特征
    features = {
        i: {
            "effect": ensure_not_numpy(shap_values[0, i]),
            "value": ""
        }
        for i in range(len(feature_names)) if shap_values[0, i] != 0
    }

    # 组织数据结构
    data = {
        "outNames": ["f(x)"],
        "baseValue": base_values,
        "outValue": np.sum(shap_values[0]) + base_values,
        "link": "identity",
        "featureNames": feature_names,
        "features": features,
        "plot_cmap": "RdBu",
    }

    # 处理正负贡献特征
    neg_features, total_neg, pos_features, total_pos = format_data(data)
    base_value = data['baseValue']
    out_value = data['outValue']
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.04

    # Compute axis limit
    # update_axis_limits(ax, total_pos, pos_features, total_neg,
    #                    neg_features, base_value, out_value)

    # def update_axis_limits(ax, total_pos, pos_features, total_neg,
    #                    neg_features, base_value, out_value):
    ax.set_ylim(-0.5, 0.15)
    padding = np.max([np.abs(total_pos) * 0.2,
                      np.abs(total_neg) * 0.2])

    if len(pos_features) > 0:
        min_x = min(np.min(pos_features[:, 0].astype(float)), base_value) - padding
    else:
        min_x = out_value - padding
    if len(neg_features) > 0:
        max_x = max(np.max(neg_features[:, 0].astype(float)), base_value) + padding
    else:
        max_x = out_value + padding
    ax.set_xlim(min_x, max_x)

    ax.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False,
                    labeltop=True, labelbottom=False)
    ax.locator_params(axis='x', nbins=12)

    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
        if key != 'top':
            spine.set_visible(False)

    # Define width of bar
    width_bar = 0.1
    text_rotation = 0
    min_perc = 0.05
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200

    # Create bar for negative shap values
    rectangle_list, separator_list = draw_bars(out_value, neg_features, 'negative',
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    # Create bar for positive shap values
    rectangle_list, separator_list = draw_bars(out_value, pos_features, 'positive',
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    # Add labels
    total_effect = np.abs(total_neg) + total_pos
    fig, ax = draw_labels(fig, ax, out_value, neg_features, 'negative',
                          offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)

    fig, ax = draw_labels(fig, ax, out_value, pos_features, 'positive',
                          offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)

    # higher lower legend
    ax.text(out_value - offset_text, 0.405, 'higher',
            fontsize=13, color='#FF0D57',
            horizontalalignment='right')

    ax.text(out_value + offset_text, 0.405, 'lower',
            fontsize=13, color='#1E88E5',
            horizontalalignment='left')

    ax.text(out_value, 0.4, r'$\leftarrow$',
            fontsize=13, color='#1E88E5',
            horizontalalignment='center')

    ax.text(out_value, 0.425, r'$\rightarrow$',
            fontsize=13, color='#FF0D57',
            horizontalalignment='center')
    # Add label for base value
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    text_out_val = ax.text(base_value, 0.33, 'base value',
                           fontsize=12, alpha=0.5,
                           horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    # Add output label
    out_name = data['outNames'][0]

    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    text_out_val = ax.text(out_value, 0.25, f'{out_value:.2f}',
                           fontsize=14, fontweight="bold",
                           horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
    # 函数f(x)
    text_out_val = ax.text(out_value, 0.33, out_name,
                           fontsize=13, alpha=0.5,
                           horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)


def lime_plot(exp, fig, save_path=None, colors=None):
    if colors is None:
        c1, c2 = '#EE8636', "#3B75AF"
    else:
        c1, c2 = colors
    predicted_value = exp.predicted_value
    min_value = exp.min_value
    max_value = exp.max_value
    axp = exp.as_list(label=1)
    vals = [x[1] for x in axp]
    names = [x[0] for x in axp]

    ax = fig.add_axes([0.05, 0.05, 0.5, 0.85])
    ax2 = fig.add_axes([0.65, 0.8, 0.32, 0.14])
    ax3 = fig.add_axes([0.62, 0.15, 0.32, 0.5])
    # 绘制水平条
    ax2.set_xlim(-0.1, 0.4)
    ax2.set_ylim(-0.2, 0.6)

    filled_width = (predicted_value - min_value) / (max_value - min_value) * 0.2
    bar_height = 0.2
    # 在 Axes 中添加一个不填充、只有黑色边框的矩形
    border_rect = patches.Rectangle((0, 0), 0.2, bar_height, fill=False, edgecolor='black', linewidth=1.5)
    ax2.add_patch(border_rect)
    filled_rect = patches.Rectangle((0.001, 0.01), filled_width, bar_height - 0.02, facecolor=c1, edgecolor='none')
    ax2.add_patch(filled_rect)

    ax2.text(-0.02, 0.1, f"{min_value:.2f} (min)", ha='right', va='center', fontsize=14)
    ax2.text(0.22, 0.1, f"{max_value:.2f} (max)", ha='left', va='center', fontsize=14)
    ax2.text(filled_width, -0.1, f"{predicted_value :.2f}", ha='center', va='top', fontsize=14, color='black')
    ax2.text(0.02, 0.4, f"Predicted vale", ha='left', va='center', fontsize=16, color='black')
    vals.reverse()
    names.reverse()
    colors = [c1 if x > 0 else c2 for x in vals]
    pos = np.arange(len(axp)) + .5
    bars = ax.barh(pos, vals, align='center', color=colors, height=0.4, edgecolor='black')

    # 生成表格需要的 cell 文本，每行 [feature, value]
    cell_text = []
    for f, v in zip(names[::-1], vals[::-1]):
        cell_text.append([f, f"{v:.2f}"])
    table = ax3.table(
        cellText=cell_text,
        # colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.6, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for key, cell in table.get_celld().items():
        cell.set_height(0.115)
        text = cell.get_text().get_text()
        color = c2 if "<=" in text or "-" in text else c1
        cell.set_facecolor(color)
    ax3.text(0.67, 1.15, f"Feature  Value", ha='center', va='center', fontsize=17, color='black')

    i = 0
    ax.text(0.1, 10.8, "positive", va='center', ha='left', fontsize=22, color=c1, fontweight="bold")
    ax.text(-0.1, 10.8, "nagative", va='center', ha='right', fontsize=22, color=c2, fontweight="bold")
    for bar, val in zip(bars, vals):
        width = bar.get_width()
        bar_y = bar.get_y() + bar.get_height() / 2
        if val > 0:
            ax.text(width + 0.01, bar_y, f"{val:.2f}", va='center', ha='left', fontsize=13)
            ax.text(0.02, bar_y + 0.5, names[i], va='center', ha='left', fontsize=15)
        else:
            ax.text(width - 0.01, bar_y, f"{val:.2f}", va='center', ha='right', fontsize=13)
            ax.text(-0.02, bar_y + 0.5, names[i], va='center', ha='right', fontsize=15)
        i += 1
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0, 10)
    for _ in [ax, ax2, ax3]:
        _.set_yticks([])
        _.set_xticks([])
        _.grid(False)
        for sp in _.spines.values():
            sp.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(2)
    # 调整上下左右边距（相对于 Figure 大小的百分比）
    # plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)
    plt.show()
