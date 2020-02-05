import os
import sys
from colors import magenta, cyan, yellow, green
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from form.file_management import file_names as fn
from scipy.stats import linregress
from form._feat_colors import color_exp


# region --- FUNCTIONS ---


def import_datasets():
    df_form = pd.read_csv(fn['scaled'] + '.csv')
    print(f'--> {green(fn["scaled"])}.csv\t\timported')

    df_wsl = pd.read_csv(fn['wsl'] + '.csv')
    print(f'--> {green(fn["wsl"])}.csv\t\timported')

    df_wsl['board_length'] = df_wsl['board_length'] * 0.0254

    return df_form, df_wsl


def concat_dfs(df_form, df_wsl, concat):
    df_form['surfer_wsl'] = 0

    if concat:
        cols_to_keep = [
            # 'surfer_name',
            # 'board_model',
            # 'board_brand',
            # 'board_shaper',
            # 'ref_date',
            # 'ref_01',
            # 'ref_02',
            'surfer_experience',
            'surfer_exercise_frequency',
            'surfer_weight',
            'surfer_height',
            # 'wave_city',
            # 'board_length-raw',
            # 'board_nose_shape-raw',
            # 'board_tail_shape-raw',
            'board_length',
            'board_width',
            'board_thickness',
            'board_nose_shape',
            'board_tail_shape',
            'board_volume',
            'surfer_gender',
            'surfer_wsl',
        ]
        df_wsl = df_wsl.loc[:, cols_to_keep]
        df_wsl = df_wsl.infer_objects()
        df_wsl['board_type'] = 4
        df_wsl['board_adequate'] = 1
        df = pd.concat([df_form, df_wsl], sort=True)

    else:
        df = df_form

    return df


def filter_dataset(df):
    '''Filters dataset - surfer_experience, board_type, surfer_weight'''
    print(f'\tNumber of samples {magenta("BEFORE")} filter: {df.shape[0]}')

    condition = df[(df['surfer_weight'] == 87) & (df['board_volume'] == 28)]
    df.drop(condition.index, inplace=True)  # drop Chumbo

    df.query('board_adequate > 0', inplace=True)  # filter adequate boards

    df.query('surfer_gender == 0', inplace=True)  # filter men
    # df.query('surfer_gender == 1', inplace=True)  # filter women

    # df.query('board_type == 4', inplace=True)  # filter pranchinha

    # df.query('surfer_experience == 0.6', inplace=True)
    # df.query('surfer_experience == 0.8', inplace=True)
    # df.query('surfer_experience == 1', inplace=True)

    # df.query('surfer_exercise_frequency < 0.5', inplace=True)
    # df.query('surfer_wsl == 1', inplace=True)
    # df.query('board_volume < 40', inplace=True)
    # df.query('74 <= surfer_weight <= 76', inplace=True)

    print(f'\tNumber of samples {magenta("AFTER")} filter: {df.shape[0]}')
    return df


def drop_features(df, size='aaa'):
    columns_medium = [
        'surfer_age',
        'board_adequate',
        'board_type',
        # 'board_volume',
        # 'board_width',
        # 'board_thickness',
        # 'board_length',
        'board_nose_shape',
        'board_tail_shape',
        'board_nose_rocker',
        'board_tail_rocker',
        # 'surfer_style',
        # 'surfer_experience',
        # 'surfer_weight',
        # 'surfer_exercise_frequency',
        # 'surfer_height',
        'surfer_gender',
        'surfer_wsl',
        # 'wave_shape',
        # 'wave_height_mean',
    ]
    columns_small = [
        'surfer_age',
        'board_adequate',
        'board_type',
        # 'board_volume',
        # 'board_width',
        # 'board_thickness',
        # 'board_length',
        'board_nose_shape',
        'board_tail_shape',
        'board_nose_rocker',
        'board_tail_rocker',
        # 'surfer_style',
        # 'surfer_experience',
        # 'surfer_weight',
        # 'surfer_exercise_frequency',
        # 'surfer_height',
        'surfer_gender',
        'surfer_wsl',
        # 'wave_shape',
        # 'wave_height_mean',
    ]

    if size == 'small':
        df.drop(columns=columns_small, inplace=True)
    else:
        df.drop(columns=columns_medium, inplace=True)

    print(f'\tDropped features: {magenta(columns_medium)}')


def filter_df_corr(df_corr):
    df_corr = df_corr.filter(regex='board', axis=0)  # rows
    df_corr = df_corr.filter(regex='board|surfer|wave', axis=1)  # columns
    print(df_corr['surfer_experience'])

    # test
    # quit()
    # df_corr = df_corr.query('surfer_experience == 0.8')
    # print(df_corr)
    # quit()
    # test

    print('\tCorrelation Matrix Filtered')

    return df_corr


def set_ticks(plt, df_corr):
    fontsize = 10

    xticks = np.arange(0.5, len(df_corr.columns), 1)
    yticks = np.arange(0.5, len(df_corr.index), 1)
    plt.xticks(xticks, df_corr.columns,
               rotation='vertical', ha='right', fontsize=fontsize)
    plt.yticks(yticks, df_corr.index, va='bottom')
    # plt.xlabel('estiloso')  # test


def annotate_matrix(ax, df_corr, round_num=2):
    df_corr = df_corr.round(round_num)
    columns = df_corr.columns
    matrix = df_corr.values
    index = df_corr.index

    for i in range(len(index)):
        for j in range(len(columns)):
            ax.text(j, i, matrix[i, j],
                    ha="center", va="center",
                    color="black", fontsize=10)
    return df_corr


def reorder_features(df):
    new_cols3 = [
        'board_adequate',
        'board_type',
        'board_volume',
        'board_width',
        'board_thickness',
        'board_length',
        'board_nose_shape',
        'board_tail_shape',
        'board_nose_rocker',
        'board_tail_rocker',
        'manoeuvres_01_paddling',
        'manoeuvres_02_drop',
        'manoeuvres_03_straight_ahead',
        'manoeuvres_04_wall_riding',
        'manoeuvres_05_floater',
        'manoeuvres_06_cut_back',
        'manoeuvres_07_rasgada',
        'manoeuvres_08_off_the_lip',
        'manoeuvres_09_tube',
        'manoeuvres_10_air',
        'performance_control',
        'performance_ease_paddling',
        'performance_flotation',
        'performance_hold',
        'performance_manoeuvrability',
        'performance_passing_through',
        'performance_stability',
        'performance_surf_speed',
        'surfer_style',
        'surfer_experience',
        'surfer_exercise_frequency',
        'surfer_weight',
        'surfer_age',
        'surfer_height',
        'surfer_gender',
        'wave_shape',
        'wave_height_mean',
        'surfer_wsl',  # test
    ]
    cols_merged = [
        'board_adequate',
        'board_type',
        'board_volume',
        'board_width',
        'board_thickness',
        'board_length',
        'board_nose_shape',
        'board_tail_shape',
        'board_nose_rocker',
        'board_tail_rocker',
        'manoeuvres_01_paddling',
        'manoeuvres_02_drop',
        'manoeuvres_03_straight_ahead',
        'manoeuvres_04_wall_riding',
        'manoeuvres_05_floater',
        'manoeuvres_06_cut_back',
        'manoeuvres_07_rasgada',
        'manoeuvres_08_off_the_lip',
        'manoeuvres_09_tube',
        'manoeuvres_10_air',
        'performance_control',
        'performance_ease_paddling',
        'performance_flotation',
        'performance_hold',
        'performance_manoeuvrability',
        'performance_passing_through',
        'performance_stability',
        'performance_surf_speed',
        'surfer_style',
        'surfer_experience',
        'surfer_weight',
        'surfer_exercise_frequency',
        'surfer_age',
        'surfer_height',
        'surfer_gender',
        'wave_shape',
        'wave_height_mean',
        'surfer_wsl',
    ]
    df = df.reindex(columns=new_cols3)
    print('\tFeatures reordered')

    return df


def set_plot(df_corr, selection):
    # set fig, ax, plt
    factor = 0.5
    fig, ax = plt.subplots()
    # plt.subplots_adjust(left=0.25, right=0.9, top=1, bottom=0.3)
    plt.title('Surfboard - correlation matrix', size=20)
    print(cyan(df_corr.shape))

    if selection == 'single':
        fig.set_size_inches(20, 4)
    else:
        width = df_corr.shape[0] * (factor + 0.2) + 2
        height = df_corr.shape[1] * factor
        # width = 10
        # height = 10
        fig.set_size_inches(width, height)
    set_ticks(plt, df_corr)

    return ax


def plot_and_save_correlation_matrix(filename, df_corr):
    ax.imshow(df_corr, cmap='coolwarm', vmin=-1, vmax=1)  # bwr

    fig_path = '_output/correlation/'
    file_path = f'{fig_path}{filename}.png'
    plt.savefig(file_path, bbox_inches="tight")
    print(f'<-- Figure {green(file_path)}\t\texported')

    root_path = os.path.dirname(sys.modules['__main__'].__file__)
    print(f"\tfile:///{root_path}/{file_path}")


def export_correlation_matrix(df_corr):
    df_corr.round(2).to_csv(fn['corr'])
    print(f'<-- Dataframe {green(fn["corr"])}.csv\t\texported')


def print_cols_and_index(df, print_index=False):
    cols = df.columns
    index = df.index
    for col in cols:
        # print(magenta(col))
        print("'" + f'{magenta(col)}' + "',")

    for idx in index:
        if print_index:
            print("'" + f'{cyan(idx)}' + "',")


def set_max_correlation_df(df_corr):
    df_max_corr = pd.DataFrame()
    cols = df_corr.columns
    index = df_corr.index
    # print(cols)
    trigger = 0.6
    for feature_idx in index:
        s_corr = df_corr.loc[feature_idx, :]
        # print(magenta(feature_idx))
        # print(yellow(s_corr))
        for feature_col, value in s_corr.iteritems():
            # print(cyan(feature_idx), value)
            if abs(value) > trigger and value != 1 and value != 0:
                s = pd.Series([feature_col, feature_idx, value, abs(value)])
                df_max_corr = df_max_corr.append(s, ignore_index=True)

    df_max_corr.columns = ['feat_col', 'feat_idx', 'value', 'abs_value']
    df_max_corr.sort_values(
        by=['feat_col', 'feat_idx'], inplace=True, ascending=False)
    print(f'\t{magenta(df_max_corr.shape[0])} correlations above {trigger}')

    return df_max_corr


def export_max_correlation_df(df_max_corr):  # fixme
    df_max_corr.to_csv(fn['top_corr'] + '.csv')
    print(f'<-- {green(fn["top_corr"])}.csv\t\texported')


def get_descriptive_statistics(df):
    # df = df.filter(regex='board', axis=1)
    # print(magenta('Mean'), f'\n{df.mean()}')
    # print(magenta('Median'), f'\n{df.median()}')
    # print(magenta('Min'), f'\n{df.min()}')
    # print(magenta('Max'), f'\n{df.max()}')

    print(df.describe().round(2).transpose().to_string())


def set_steps_to_plot():
    steps = [
        ('surfer_weight', 'board_volume'),

        ('board_volume', 'board_width'),
        ('surfer_style', 'board_width'),
        ('surfer_experience', 'board_width'),
        ('surfer_exercise_frequency', 'board_width'),
        ('wave_shape', 'board_width'),

        ('board_volume', 'board_thickness'),
        ('board_width', 'board_thickness'),
        ('surfer_style', 'board_thickness'),
        ('wave_shape', 'board_thickness'),

        ('board_volume', 'board_length'),
        ('board_width', 'board_length'),
        ('board_thickness', 'board_length'),
        ('surfer_style', 'board_length'),
        ('surfer_height', 'board_length'),
        ('wave_height_mean', 'board_length'),

        ('wave_height_mean', 'board_volume'),
    ]
    print(f'{magenta(len(steps))} steps to plot')

    return steps


def get_feat_lims(x, y):
    feat_lims = {
        'board_volume': (20, 40),
        'board_width': (17.5, 22.5),
        'board_thickness': (1.9, 3.1),
        'board_length': (1.5, 2.2),
        'surfer_style': (-1.5, 1.5),  # label dict
        'surfer_experience': (0.4, 1.2),
        'surfer_exercise_frequency': (-0.2, 1.2),
        'surfer_weight': (50, 100),
        'surfer_height': (1.5, 2.2),
        'wave_shape': (-1.2, 1.2),
        'wave_height_mean': (1, 7),
    }

    xlim = feat_lims[x]
    ylim = feat_lims[y]

    # return feat_lims
    return xlim, ylim


def set_plot_name_and_save(pair, kind, s_exp, s=None):
    if len(pair) == 4:
        x, y, i, j = pair
        reg_name = f'{i}-{j}_{y}-{x}'
        full_path = f'_output/correlation/{kind}/{reg_name}.png'

    else:
        x, y = pair
        if s_exp:
            reg_name = f's{s}_{s_exp}_{y}-{x}'
        else:
            reg_name = f's{s}_{y}-{x}'
        full_path = f'_output/steps/{reg_name}.png'

    plt.savefig(full_path, bbox_inches="tight")
    print(f'<-- {green(full_path)}\t\texported')

    root_path = os.path.dirname(sys.modules['__main__'].__file__)
    print(f"\tfile:///{root_path}/{full_path}")
    print(kind, magenta(reg_name))


def annotate_plot(ax, x=None, y=None, s_exp=None, df=None, i_s=None):
    x_center = 0.448

    annotate_units(ax, x, y)
    annotate_feat_labels(ax, x, x_center)
    annotate_guideline(ax, x_center, i_s)
    if s_exp is not 'all':
        annotate_reg_equation(ax, df, s_exp, x, y)


def annotate_units(ax, x, y):
    feat_units = {
        'board_volume': 'litros',
        'board_width': 'polegadas',
        'board_thickness': 'polegadas',
        'board_length': 'm',
        'surfer_weight': 'kg',
        'surfer_height': 'm',
        'wave_height_mean': 'm',
    }
    if x in feat_units.keys():
        # ax.set_axis_labels(xlabel=f'{x} [{feat_units[x]}]')
        ax.ax_joint.set_xlabel(xlabel=f'{x} [{feat_units[x]}]')

    if y in feat_units.keys():
        # ax.set_axis_labels(ylabel=f'{y} [{feat_units[y]}]')
        ax.ax_joint.set_ylabel(ylabel=f'{y} [{feat_units[y]}]')

    return ax


def annotate_feat_labels(ax, x, x_center):
    feat_labels = {
        'surfer_style':
            '(-1) Suave | (0) Normal | (+1) Agressivo',
        'surfer_experience':
            '(0.2) Aprendiz | (0.4) Iniciante | (0.6) Intermediário | (0.8) Avançado | (1.0) Profissional',
        'wave_shape':
            '(-1) Cheia | (0) Meio cavada | (+1) Cavada',
    }
    if x in feat_labels.keys():
        ax.fig.text(x_center, -0.015, feat_labels[x], fontsize=10,
                    horizontalalignment='center', verticalalignment='bottom')

    return ax


def annotate_guideline(ax, x_center, i_s):
    df_guidelines = pd.read_csv('_input/df_guidelines.csv')
    guidelines = df_guidelines['guidelines'].tolist()
    guideline = f'#{i_s}: "{guidelines[i_s]}"'
    print(repr(guideline))
    guideline = guideline.replace('\r', '')
    print(repr(guideline))
    # quit()

    ax.fig.text(x_center, -0.10, guideline,
                fontsize=11,
                horizontalalignment='center', verticalalignment='bottom')


def annotate_reg_equation(ax, df, s_exp, x, y):
    color = 'black'
    if type(s_exp) is not str:
        df = df.query(f'surfer_experience == {s_exp}')
        color = color_exp[s_exp]

    df_c = df.copy()
    df_c.dropna(inplace=True, subset=[x, y])
    slope, intercept, r_value, p_value, std_err = linregress(df_c[x], df_c[y])
    variance = std_err ** 2
    r_square = r_value ** 2

    to_annotate = f'y = {slope:.3f}x + {intercept:.2f}\n' \
        f'r:{r_value:.2f} p:{p_value:.2g} s:{std_err:.3f}\n' \
        f'v:{variance:.4f} r2:{r_square:.2f} '

    ax.fig.text(0.8, 0.1, to_annotate,
                fontsize=12, color=color,
                horizontalalignment='right', verticalalignment='bottom')


def add_legend(plt, experience):
    num_samples = count_df_queries(df)

    from matplotlib.lines import Line2D as line2d
    handles = [
        line2d([], [], color='gray', label='Outros'),
        line2d([], [], color=color_exp[0.6],
               label=f'Intermediário ({num_samples[0.6]})'),
        line2d([], [], color=color_exp[0.8],
               label=f'Avançado ({num_samples[0.8]})'),
        line2d([], [], color=color_exp[1.0],
               label=f'Profissional ({num_samples[1.0]})'),
        line2d([], [], color='gray', label='Todos'),
    ]

    if experience == 0.6:
        handles = handles[:2]
    elif experience == 0.8:
        handles = [handles[0], handles[2]]
    elif experience == 1.0:
        handles = [handles[0], handles[3]]
    elif experience is 'no_filter':
        handles = [handles[4]]
    else:
        handles = handles[1:4]

    plt.legend(handles=handles, title='Nível de experiência:')


def count_df_queries(df):
    num_samples = {
        0.6: df.query('surfer_experience == 0.6').shape[0],
        0.8: df.query('surfer_experience == 0.8').shape[0],
        1.0: df.query('surfer_experience == 1.0').shape[0],
        'all': df.shape[0],
    }
    print(magenta(num_samples))

    return num_samples


def plot_jointplot(df, pair, kind='kde', i_s=None, color_exp=color_exp):
    if len(pair) == 4:
        x, y, i, j = pair
    else:
        x, y = pair

    sns.set(color_codes=True)
    plt.clf()

    # s_exp = 'no_filter'
    # s_exp = 0.6
    # s_exp = 0.8
    # s_exp = 1.0
    s_exp = 'all'

    if s_exp == 'all':
        reg_list = [
            {'data': df, 'color': 'gray', 'fit_reg': False},
            {'data': df.query('surfer_experience == 1.0'),
             'color': color_exp[1.0]},
            {'data': df.query('surfer_experience == 0.8'),
             'color': color_exp[0.8]},
            {'data': df.query('surfer_experience == 0.6'),
             'color': color_exp[0.6]},
        ]
        data = df
        color = 'gray'
        ci = None

    elif s_exp == 'no_filter':
        reg_list = [
            {'data': df, 'color': 'gray', 'fit_reg': True},
        ]
        data = df
        color = 'gray'
        ci = 95

    else:
        reg_list = [
            {'data': df, 'color': 'gray', 'fit_reg': False},
            {'data': df.query(f'surfer_experience == {s_exp}'),
             'color': color_exp[s_exp]},
        ]
        data = df.query(f'surfer_experience == {s_exp}')
        color = color_exp[s_exp]
        ci = 95

    xlim, ylim = get_feat_lims(x, y)
    g = sns.jointplot(x, y, data=data, color=color, xlim=xlim, ylim=ylim,
                      kind='reg', dropna=True, fit_reg=False,
                      marginal_kws=dict(bins=9), height=8, )

    reg_kwargs_main = {
        'x': x,
        'y': y,
        'scatter': True,
        'ax': g.ax_joint,
        'truncate': False,
        'ci': ci,
        'x_jitter': 0.05,
    }

    for reg_kwargs_var in reg_list:
        ax_reg = sns.regplot(**reg_kwargs_main, **reg_kwargs_var)

    annotate_plot(g, x=x, y=y, s_exp=s_exp, df=df, i_s=i_s)
    add_legend(plt, s_exp)
    set_plot_name_and_save(pair, kind, s_exp, i_s)
    # plt.show()  # test
    plt.close()


def plot_loop(plot_type='kde'):
    new_cols = df_corr.columns
    new_index = df_corr.index

    for i, row in enumerate(new_index):
        for j, col in enumerate(new_cols):
            a = df_corr.values[i][j]
            pair = (col, row, i, j)
            print(pair)
            if pair[0] != pair[1]:
                if not np.isnan(a):
                    print('inside if')
                    plot_jointplot(df, pair, kind=plot_type)


def check_discreteness(df, feat_x, feat_y):
    len_unique_x = len(df[feat_x].unique())
    len_unique_y = len(df[feat_y].unique())
    if len_unique_x < 10 and len_unique_y < 10:
        discreteness = True
        print(magenta('Features are discrete'))
    else:
        discreteness = False

    return discreteness


def get_cross_tab(df, x, y):
    if check_discreteness(df, x, y):
        ct = pd.crosstab(df[x], df[y])
        ct.sort_index(ascending=False, inplace=True)
        print('\n', '-' * 80, '\n', ct)


def plot_pairgrid(df):
    pairs = set_steps_to_plot()
    print(pairs)
    x_vars = [
        'board_volume',
        'board_width',
        'board_thickness',
        'board_length',
        'surfer_style',
        'surfer_experience',
        'surfer_exercise_frequency',
        'surfer_weight',
        'surfer_height',
        'wave_shape',
        'wave_height_mean',
    ]
    y_vars = [
        'board_volume',
        'board_width',
        'board_thickness',
        'board_length',
    ]

    g = sns.PairGrid(df, x_vars=x_vars, y_vars=y_vars, dropna=True)
    g.map(plt.scatter)
    plt.savefig('_output/multiples/board_width.png', bbox_inches="tight")


# endregion

# --- PIPELINE ---
# region Prepare Dataset

df_form, df_wsl = import_datasets()

# test board_adequate.value_counts()
print(df_form.board_adequate.value_counts())
""" 0.5    79
 0.0    38
 1.0    37
-0.5     5
-1.0     5

Total 164
"""


# quit()
# test

df = concat_dfs(df_form, df_wsl, concat=True)
df = reorder_features(df)
filter_dataset(df)
drop_features(df)

# endregion
# region Prepare Correlation Matrix

df_corr = df.corr()
# df_corr = filter_df_corr(df_corr)
df_max_corr = set_max_correlation_df(df_corr)
export_max_correlation_df(df_max_corr)

# endregion
# region Plot correlation matrix
ax = set_plot(df_corr, selection='all')
annotate_matrix(ax, df_corr, round_num=2)
plot_and_save_correlation_matrix(f'matrices/simple', df_corr)
export_correlation_matrix(df_corr)

# endregion
# region Plot plot plot

# plot_jointplot(df, (x, y, 00, 0), kind='kde')
# plot_loop(plot_type='reg')
# plot_pairgrid(df)

# endregion
# region Plot specification steps
steps = set_steps_to_plot()
i_s = 0
plot_jointplot(df, steps[i_s], i_s=i_s, kind='kde')

# for i_s, step in enumerate(steps):
#     plot_jointplot(df, step, i_s=i_s, kind='kde')
#
#     if i_s == 0:
#         break

# endregion
# region Get main statistics

# get_cross_tab(df, x, y)

# get_descriptive_statistics(df)

# endregion
