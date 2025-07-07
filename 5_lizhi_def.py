import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================= 1. 系统参数定义 =============================
def define_system_parameters():
    """定义系统参数：节点、发电机、负荷和线路数据"""
    nodes = [1, 2, 3, 4, 5]

    # 发电机数据: {节点: (最大出力[GW], 成本[$/MWh])}
    generators = {
        1: (17.335, 407),  # 节点1发电机
        3: (9.611, 506),  # 节点3发电机
        4: (1.775, 442.36)  # 节点4发电机
    }

    # 负荷数据: {节点: 负荷值[GW]}
    loads = {
        2: 11.156,  # 节点2负荷
        3: 11.181,  # 节点3负荷
        5: 6.384  # 节点5负荷
    }

    # 线路数据: (起点, 终点, 电抗, 容量[GW])
    branches = [
        (1, 5, 0.1, 12.0),  # 线路1-5
        (1, 4, 0.1, 3.5),  # 线路1-4
        (1, 2, 0.1, 9.2),  # 线路1-2
        (2, 3, 0.1, 5.2),  # 线路2-3
        (3, 4, 0.1, 5.7),  # 线路3-4
        (4, 5, 0.1, 2.0)  # 线路4-5
    ]

    return nodes, generators, loads, branches


# ============================= 2. 创建优化模型 =============================
def create_optimization_model(nodes, generators, loads, branches):
    """创建并配置优化模型"""
    model = pyo.ConcreteModel()

    # 定义集合
    model.NODES = pyo.Set(initialize=nodes)
    model.BRANCHES = pyo.Set(initialize=range(len(branches)))

    # 定义变量
    # - `model.theta`：节点电压相角（弧度），有上下界（-100到100）
    # - `model.Pg`：发电机出力（非负实数），对于没有发电机的节点，上界设为0（即不能发电）
    # - `model.Pf`：线路潮流，有上下界（-300到300，但后面有线路容量约束，所以这个界可以设大一些，
    # 但为了求解效率，我们根据线路容量约束来设置，但这里我们使用一个较大的值）
    model.theta = pyo.Var(model.NODES, bounds=(-100, 100), initialize=0)
    model.Pg = pyo.Var(model.NODES, within=pyo.NonNegativeReals, initialize=0)

    # 设置发电机出力上限
    for n in model.NODES:
        if n in generators:
            model.Pg[n].setub(generators[n][0])
        else:
            model.Pg[n].setub(0)

    model.Pf = pyo.Var(model.BRANCHES, bounds=(-300, 300), initialize=0)

    # 目标函数
    def cost_rule(model):
        return sum(model.Pg[n] * generators[n][1] for n in model.NODES if n in generators)

    model.cost = pyo.Objective(rule=cost_rule, sense=pyo.minimize)

    # 参考节点约束
    model.ref_bus = pyo.Constraint(expr=model.theta[1] == 0)

    # 节点功率平衡约束
    def power_balance_rule(model, n):
        gen_power = model.Pg[n]
        load_power = loads[n] if n in loads else 0
        branch_flow = 0

        for idx, (i, j, x, limit) in enumerate(branches):
            if i == n:
                branch_flow += model.Pf[idx]
            if j == n:
                branch_flow -= model.Pf[idx]

        return gen_power - load_power == branch_flow

    model.balance = pyo.Constraint(model.NODES, rule=power_balance_rule)

    # 线路潮流约束
    def branch_flow_rule(model, idx):
        i, j, x, limit = branches[idx]
        return model.Pf[idx] == (model.theta[i] - model.theta[j]) / x

    model.branch_flow = pyo.Constraint(model.BRANCHES, rule=branch_flow_rule)

    # 线路容量约束
    model.branch_limit = pyo.Constraint(model.BRANCHES, rule=lambda m, idx:
    (-branches[idx][3], m.Pf[idx], branches[idx][3]))

    # 启用对偶变量记录
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return model


# ============================= 3. 求解函数 =============================
def solve_model(model, with_constraints=False):
    """求解优化模型"""
    solver = pyo.SolverFactory('highs')

    if not with_constraints:
        print("\n===== 无容量约束求解 =====")
        model.branch_limit.deactivate()  # 禁用容量约束
    else:
        print("\n===== 带容量约束求解 =====")
        model.branch_limit.activate()  # 启用容量约束

    results = solver.solve(model, tee=True)
    return results


# ============================= 4. 结果分析函数 =============================
def analyze_results(model, branches, generators, loads, results, with_constraints=False):
    """分析求解结果并返回关键指标"""
    if results.solver.status != pyo.SolverStatus.ok:
        print("\n求解失败!")
        return None

    # 加载解到模型
    model.solutions.load_from(results)

    # 计算总成本
    total_cost = pyo.value(model.cost)
    print(f"\n求解成功! 总发电成本: ${total_cost:.2f}")

    # 收集发电机出力
    Pg_values = {}
    for n in model.NODES:
        if n in generators:
            Pg_values[n] = pyo.value(model.Pg[n])

    # 收集节点相角
    theta_values = {n: pyo.value(model.theta[n]) for n in model.NODES}

    # 收集线路潮流
    Pf_values = {}
    for idx in model.BRANCHES:
        Pf_values[idx] = pyo.value(model.Pf[idx])

    # 检查线路潮流是否超限
    overloads = []
    for idx, (i, j, x, limit) in enumerate(branches):
        flow = Pf_values[idx]
        if abs(flow) > limit:
            overloads.append((i, j, flow, limit, abs(flow) - limit))

    # 计算LMP
    lmp_values = {}
    for n in model.NODES:
        try:
            lmp_values[n] = model.dual[model.balance[n]]
        except KeyError:
            lmp_values[n] = None

    # 返回分析结果
    analysis = {
        'total_cost': total_cost,
        'Pg': Pg_values,
        'theta': theta_values,
        'Pf': Pf_values,
        'lmp': lmp_values,
        'overloads': overloads
    }

    return analysis


# ============================= 5. 可视化函数 =============================
def visualize_results(nodes, generators, loads, branches, unconstrained_analysis, constrained_analysis):
    """可视化分析结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(14, 10))

    # 创建网络图
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)

    for i, j, x, limit in branches:
        G.add_edge(i, j, weight=limit)

    # 节点位置布局
    pos = {1: (0, 1), 2: (1, 0), 3: (1, 2), 4: (2, 1), 5: (3, 1)}

    # 1. 系统拓扑图
    plt.subplot(221)
    nx.draw(G, pos, with_labels=False, node_size=1500, node_color='lightblue', font_size=10, font_weight='bold')

    # 添加节点标签
    node_labels = {}
    for n in nodes:
        if n in generators:
            node_labels[
                n] = f"节点{n}\nG{n}最大出力:{generators[n][0]}MW\n成本:{generators[n][1]}$/MWh"
        elif n in loads:
            node_labels[n] = f"节点{n}\n负荷:{loads[n]}MW"
        else:
            node_labels[n] = f"节点{n}"

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)

    # 添加线路标签
    edge_labels = {(i, j): f"X={x}pu\n容量±{limit}MW" for i, j, x, limit in branches}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("5节点电力系统拓扑")

    # 2. LMP比较图
    plt.subplot(222)
    nodes_sorted = sorted(nodes)
    lmp_unc = [unconstrained_analysis['lmp'].get(n, 0) for n in nodes_sorted]
    lmp_con = [constrained_analysis['lmp'].get(n, 0) for n in nodes_sorted]
    bar_width = 0.35
    x = np.arange(len(nodes_sorted))

    # 添加数值标签
    for i, v in enumerate(lmp_con):
        plt.text(x[i] + bar_width / 2, v + 0.5, f"{v:.1f}", ha='center')
    for i, v in enumerate(lmp_unc):
        plt.text(x[i] - bar_width / 2, v + 0.5, f"{v:.1f}", ha='center')

    plt.bar(x - bar_width / 2, lmp_unc, bar_width, label='无约束LMP')
    plt.bar(x + bar_width / 2, lmp_con, bar_width, label='约束LMP')
    plt.xlabel('节点')
    plt.ylabel('LMP (/MWh)')
    plt.title('节点边际电价比较')
    plt.xticks(x, nodes_sorted)
    plt.legend()

    # 3. 发电机出力图
    plt.subplot(223)
    gen_nodes = [n for n in nodes if n in generators]
    gen_power = [constrained_analysis['Pg'][n] for n in gen_nodes]
    gen_max = [generators[n][0] for n in gen_nodes]
    remaining = [max - act for max, act in zip(gen_max, gen_power)]

    # 绘制堆叠柱状图
    bars1 = plt.bar(gen_nodes, gen_power, width=0.6, label='实际出力', color='#1f77b4')
    bars2 = plt.bar(gen_nodes, remaining, width=0.6, bottom=gen_power,
                    label='剩余容量', alpha=0.3, color='#ff7f0e')

    # 添加数值标签
    for bar, power, max_power in zip(bars1, gen_power, gen_max):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.,
                 height / 2,
                 f'{power:.1f}/{max_power}MW',
                 ha='center',
                 va='center',
                 color='white',
                 fontweight='bold',
                 fontsize=9)

    # 添加总容量标签
    for bar, power, remain in zip(bars2, gen_power, remaining):
        if remain > 0:  # 只在有剩余容量时显示
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     power + remain / 2,
                     f'{remain:.1f}',
                     ha='center',
                     va='center',
                     color='black',
                     fontsize=8)

    plt.xlabel('节点', fontweight='bold')
    plt.ylabel('出力 (MW)', fontweight='bold')
    plt.title('发电机出力（实际/最大容量）', fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, max(gen_max) * 1.2)  # 设置y轴上限

    # 4. 线路潮流图
    plt.subplot(224)
    line_labels = [f"{i}-{j}" for i, j, x, limit in branches]
    flows = [constrained_analysis['Pf'][idx] for idx in range(len(branches))]
    capacities = [limit for i, j, x, limit in branches]

    # 将潮流分为正向和负向
    positive_flows = [f if f > 0 else 0 for f in flows]
    negative_flows = [f if f < 0 else 0 for f in flows]

    # 绘制柱状图（分开正向和负向）
    pos_bars = plt.bar(line_labels, positive_flows, color='blue', label='正向潮流')
    neg_bars = plt.bar(line_labels, negative_flows, color='orange', label='负向潮流')

    # 添加容量限制线
    plt.plot(line_labels, capacities, 'r--', label='正向容量')
    plt.plot(line_labels, [-c for c in capacities], 'r-', label='负向容量')

    # 在柱子上添加数值标签
    for bar in pos_bars + neg_bars:
        height = bar.get_height()
        if height != 0:  # 只显示非零值
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     height / 2,
                     f'{height:.1f}',
                     ha='center',
                     va='center',
                     color='white',
                     fontweight='bold')

    # 在容量线上添加数值标签
    for i, cap in enumerate(capacities):
        plt.text(i, cap + 5, f'{cap}', ha='center', color='red')
        plt.text(i, -cap - 5, f'{-cap}', ha='center', color='red')

    plt.xlabel('线路')
    plt.ylabel('潮流 (MW)')
    plt.title('线路潮流分布（正向/负向分开显示）')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('5_node_lmp_analysis.png', dpi=300)
    plt.show()

    print("\n分析完成! 结果已保存到 '5_node_lmp_analysis.png'")


# ============================= 6. 主函数 =============================
def main():
    """主函数，协调整个优化和分析流程"""
    # 1. 定义系统参数
    nodes, generators, loads, branches = define_system_parameters()

    # 2. 创建优化模型
    model = create_optimization_model(nodes, generators, loads, branches)

    # 3. 无约束求解和分析
    results_unc = solve_model(model, with_constraints=False)
    unconstrained_analysis = analyze_results(model, branches, generators, loads, results_unc)

    if unconstrained_analysis is None:
        print("无约束求解失败，终止程序")
        return

    # 4. 约束求解和分析
    results_con = solve_model(model, with_constraints=True)
    constrained_analysis = analyze_results(model, branches, generators, loads, results_con, with_constraints=True)

    if constrained_analysis is None:
        print("约束求解失败，使用无约束解")
        constrained_analysis = unconstrained_analysis

    # 5. 打印关键结果
    print("\n===== 最终优化结果 =====")
    print(f"总发电成本: ${constrained_analysis['total_cost']:.2f}")

    print("\n发电机出力:")
    for n in nodes:
        if n in generators:
            p = constrained_analysis['Pg'].get(n, 0)
            print(f"节点{n}: {p:.1f} MW (最大 {generators[n][0]} MW)")

    print("\n节点相角 (弧度):")
    for n in nodes:
        print(f"节点{n}: {constrained_analysis['theta'].get(n, 0):.4f}")

    print("\n线路潮流:")
    for idx, (i, j, x, limit) in enumerate(branches):
        flow = constrained_analysis['Pf'].get(idx, 0)
        status = "OK" if abs(flow) <= limit else "超载!"
        print(f"线路 {i}-{j}: {flow:.1f} MW | 容量 ±{limit} MW | {status}")

    # 6. 可视化结果
    visualize_results(nodes, generators, loads, branches, unconstrained_analysis, constrained_analysis)


# ============================= 执行主函数 =============================
if __name__ == "__main__":
    main()