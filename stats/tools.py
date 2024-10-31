# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_test_res(x, y, feature_name, success_num, total_size, alpha=0.05):
    """
    用于画趋势图
    """
    alpha = alpha
    plt.figure(figsize=(10, 4), dpi=120)

    plt.plot(x, y, color='g', marker='o')
    plt.plot(x, [alpha] * len(y), color='firebrick', marker='o', linestyle='dashed')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    plt.xlabel('date')
    plt.ylabel('p-value')
    plt.rcParams['font.sans-serif'] = ['Ubuntu']
    plt.title(label='{}: pass rate: {:.2%}'.format(feature_name, success_num / total_size))
    plt.show()


def run(df, print_total_log=False, print_daily_log=False, plot=False, multi_group=False, key_check_columns=[],
        alpha=0.05):
    """
    Args:
        df: 输入数据
        print_log: 是否打印日志
        plot: 是否画图
        key_check_columns: 需要检验的指标
    Return:
        res: list: [比例指标检验通过率,比例指标平均p值, 均值指标检验通过率,均值指标平均p值]
    """
    check_columns = all_check_columns if len(key_check_columns) == 0 else key_check_columns
    ratio_check_columns, average_check_columns = [i for i in check_columns if 'rate' in i], [i for i in check_columns if
                                                                                             'average' in i]
    if print_total_log:
        print("The indicator should be tested are: {}, ratio indicator num: {}, mean indicator num: {}".format(
            ','.join(check_columns), len(ratio_check_columns), len(average_check_columns)))

    ratio_success_rate, ratio_p, average_success_rate, average_p = 0, 0, 0, 0
    for test_column in check_columns:

        if test_column in average_check_columns:
            var_column, n_column = check_dict[test_column]
            # 均值指标检验
            p_res, success_num = generate_hypothesis_testing_result(df,
                                                                    test_column=test_column,
                                                                    n_column=n_column,
                                                                    m_column=None,
                                                                    var_column=var_column,
                                                                    alpha=alpha,
                                                                    print_log=print_daily_log,
                                                                    multi_group=multi_group)
            if print_total_log:
                print("Test {}, total verify {x} days, success rate: {s:.2%}.".format(test_column, x=len(p_res),
                                                                                      s=success_num / len(p_res)))
            average_success_rate += success_num / len(p_res)
            average_p += np.mean(p_res)
        elif test_column in ratio_check_columns:
            # 比例指标检验
            m_column, n_column = check_dict[test_column]
            p_res, success_num = generate_hypothesis_testing_result(df,
                                                                    test_column=test_column,
                                                                    n_column=n_column,
                                                                    m_column=m_column,
                                                                    var_column=None,
                                                                    alpha=alpha,
                                                                    print_log=print_daily_log,
                                                                    multi_group=multi_group)
            if print_total_log:
                print("Test {}, total verify {x} days, success rate: {s:.2%}.".format(test_column, x=len(p_res),
                                                                                      s=success_num / len(p_res)))
            ratio_success_rate += success_num / len(p_res)
            ratio_p += np.mean(p_res)

        # 画图
        if plot:
            # preprocessing date format
            dates = df.date.unique()
            dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in dates]
            plot_test_res(dates, p_res, test_column, success_num, total_size=len(p_res))

    return [ratio_success_rate / len(ratio_check_columns), ratio_p / len(ratio_check_columns),
            average_success_rate / len(average_check_columns), average_p / len(average_check_columns)]
