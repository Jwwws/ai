"""
十五数码问题求解器 - 完整实现
作者：AI助手
功能：使用A*算法求解十五数码问题，包含可视化、性能分析和交互式界面
"""

import heapq
import time
import math
from copy import deepcopy
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import sys
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==================== 节点类 ====================

class Node:
    """节点类，表示十五数码问题的一个状态"""

    def __init__(self, state: List[List[int]], parent=None, move=None, g=0, h=0):
        """
        初始化节点

        Args:
            state: 4x4的二维列表，表示当前状态，0代表空白
            parent: 父节点
            move: 从父节点到当前节点的移动方向
            g: 从起始节点到当前节点的实际代价
            h: 从当前节点到目标节点的启发式估计代价
        """
        self.state = state
        self.parent = parent
        self.move = move  # 移动方向：'U', 'D', 'L', 'R'
        self.g = g  # 路径成本
        self.h = h  # 启发式估计
        self.f = g + h  # 总成本

    def __lt__(self, other):
        """用于堆排序，比较节点的f值"""
        return self.f < other.f or (self.f == other.f and self.h < other.h)

    def __eq__(self, other):
        """判断两个节点状态是否相同"""
        if not isinstance(other, Node):
            return False
        return self.state == other.state

    def __hash__(self):
        """将状态转换为元组以便哈希"""
        return hash(tuple(tuple(row) for row in self.state))

    def get_blank_position(self) -> Tuple[int, int]:
        """获取空白格子的位置"""
        for i in range(4):
            for j in range(4):
                if self.state[i][j] == 0:
                    return i, j
        return None


# ==================== 十五数码求解器类 ====================

class FifteenPuzzleSolver:
    """十五数码问题求解器"""

    # 目标状态
    GOAL_STATE = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]

    def __init__(self, initial_state: List[List[int]], heuristic_type='manhattan'):
        """
        初始化求解器

        Args:
            initial_state: 初始状态
            heuristic_type: 启发函数类型 ('manhattan', 'misplaced' 或 'euclidean')
        """
        self.initial_state = initial_state
        self.heuristic_type = heuristic_type
        self.nodes_expanded = 0  # 扩展节点数
        self.nodes_generated = 0  # 生成节点数
        self.solution_path = []  # 解决方案路径
        self.solution_moves = []  # 解决方案移动序列
        self.runtime = 0  # 运行时间
        self.max_depth = 0  # 最大搜索深度

    def is_solvable(self, state: List[List[int]]) -> bool:
        """
        判断给定的十五数码状态是否可解
        基于逆序数的奇偶性判断
        """
        # 将4x4矩阵展平为一维列表，排除空白（0）
        flat_state = []
        for i in range(4):
            for j in range(4):
                if state[i][j] != 0:
                    flat_state.append(state[i][j])

        # 计算逆序数
        inversions = 0
        for i in range(len(flat_state)):
            for j in range(i + 1, len(flat_state)):
                if flat_state[i] > flat_state[j]:
                    inversions += 1

        # 找到空白格所在行（从下往上数，从1开始）
        blank_row = 0
        for i in range(4):
            for j in range(4):
                if state[i][j] == 0:
                    blank_row = 4 - i  # 从底部计数的行数
                    break

        # 对于4x4的十五数码：
        # 如果空白在从底部计数的偶数行，逆序数需为奇数才可解
        # 如果空白在从底部计数的奇数行，逆序数需为偶数才可解
        if (blank_row % 2 == 0) and (inversions % 2 == 1):
            return True
        elif (blank_row % 2 == 1) and (inversions % 2 == 0):
            return True
        else:
            return False

    def heuristic(self, state: List[List[int]]) -> float:
        """计算启发式函数值"""
        if self.heuristic_type == 'manhattan':
            return self.manhattan_distance(state)
        elif self.heuristic_type == 'misplaced':
            return self.misplaced_tiles(state)
        elif self.heuristic_type == 'euclidean':
            return self.euclidean_distance(state)
        else:
            raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")

    def manhattan_distance(self, state: List[List[int]]) -> int:
        """
        曼哈顿距离启发函数
        计算每个数字当前位置到目标位置的曼哈顿距离之和
        """
        distance = 0
        for i in range(4):
            for j in range(4):
                if state[i][j] != 0:  # 忽略空白格
                    value = state[i][j]
                    # 计算目标位置
                    target_row = (value - 1) // 4
                    target_col = (value - 1) % 4
                    # 计算曼哈顿距离
                    distance += abs(i - target_row) + abs(j - target_col)
        return distance

    def misplaced_tiles(self, state: List[List[int]]) -> int:
        """
        不在位元素个数启发函数
        计算不在正确位置上的数字个数（不包括空白格）
        """
        count = 0
        for i in range(4):
            for j in range(4):
                if state[i][j] != 0:  # 忽略空白格
                    if state[i][j] != self.GOAL_STATE[i][j]:
                        count += 1
        return count

    def euclidean_distance(self, state: List[List[int]]) -> float:
        """
        欧几里得距离启发函数
        计算每个数字当前位置到目标位置的欧几里得距离之和
        """
        distance = 0.0
        for i in range(4):
            for j in range(4):
                if state[i][j] != 0:  # 忽略空白格
                    value = state[i][j]
                    # 计算目标位置
                    target_row = (value - 1) // 4
                    target_col = (value - 1) % 4
                    # 计算欧几里得距离
                    distance += math.sqrt((i - target_row) ** 2 + (j - target_col) ** 2)
        return distance

    def get_neighbors(self, node: Node) -> List[Node]:
        """获取当前节点的所有可能后继节点"""
        neighbors = []
        i, j = node.get_blank_position()

        # 定义四个方向的移动
        moves = [
            ('U', i - 1, j),  # 上
            ('D', i + 1, j),  # 下
            ('L', i, j - 1),  # 左
            ('R', i, j + 1)  # 右
        ]

        for move, new_i, new_j in moves:
            # 检查移动是否合法
            if 0 <= new_i < 4 and 0 <= new_j < 4:
                # 创建新状态
                new_state = deepcopy(node.state)
                # 交换空白格和相邻格子
                new_state[i][j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[i][j]

                # 创建新节点
                new_node = Node(
                    state=new_state,
                    parent=node,
                    move=move,
                    g=node.g + 1,
                    h=self.heuristic(new_state)
                )
                neighbors.append(new_node)
                self.nodes_generated += 1

        return neighbors

    def a_star_search(self) -> Tuple[bool, Optional[Node]]:
        """执行A*搜索算法"""
        start_time = time.time()

        # 检查是否可解
        if not self.is_solvable(self.initial_state):
            print("该初始状态无解！")
            self.runtime = time.time() - start_time
            return False, None

        # 初始化起始节点
        start_node = Node(
            state=self.initial_state,
            parent=None,
            move=None,
            g=0,
            h=self.heuristic(self.initial_state)
        )
        self.nodes_generated = 1

        # 初始化开放列表（优先队列）和关闭列表
        open_list = []
        closed_set = set()

        # 将起始节点加入开放列表
        heapq.heappush(open_list, start_node)

        while open_list:
            # 获取f值最小的节点
            current_node = heapq.heappop(open_list)

            # 检查是否达到目标状态
            if current_node.state == self.GOAL_STATE:
                self.runtime = time.time() - start_time
                self.nodes_expanded = len(closed_set)
                return True, current_node

            # 将当前节点加入关闭列表
            current_state_hash = hash(current_node)
            if current_state_hash in closed_set:
                continue
            closed_set.add(current_state_hash)

            # 扩展当前节点
            neighbors = self.get_neighbors(current_node)
            self.nodes_expanded += 1

            # 更新最大深度
            self.max_depth = max(self.max_depth, current_node.g)

            for neighbor in neighbors:
                # 如果邻居节点已经在关闭列表中，跳过
                if hash(neighbor) in closed_set:
                    continue

                # 检查开放列表中是否已有该状态，如果有且代价更高，则更新
                found_in_open = False
                for i, open_node in enumerate(open_list):
                    if open_node.state == neighbor.state:
                        found_in_open = True
                        if neighbor.g < open_node.g:
                            # 更新节点的代价
                            open_list[i] = neighbor
                            heapq.heapify(open_list)
                        break

                # 如果不在开放列表中，加入
                if not found_in_open:
                    heapq.heappush(open_list, neighbor)

        self.runtime = time.time() - start_time
        return False, None

    def solve(self) -> bool:
        """求解十五数码问题"""
        success, goal_node = self.a_star_search()

        if success:
            # 回溯得到解决方案路径
            self.solution_path = []
            self.solution_moves = []
            current = goal_node

            while current.parent is not None:
                self.solution_path.append(current.state)
                self.solution_moves.append(current.move)
                current = current.parent

            # 添加初始状态并反转列表
            self.solution_path.append(current.state)
            self.solution_path.reverse()
            self.solution_moves.reverse()

            return True
        else:
            return False

    def print_statistics(self):
        """打印求解统计信息"""
        print("\n" + "=" * 60)
        print("求解统计信息")
        print("=" * 60)
        print(f"初始状态是否可解: {'是' if self.is_solvable(self.initial_state) else '否'}")
        print(f"是否找到解决方案: {'是' if self.solution_path else '否'}")
        print(f"扩展节点数: {self.nodes_expanded}")
        print(f"生成节点数: {self.nodes_generated}")
        print(f"解决方案步数: {len(self.solution_moves)}")
        print(f"最大搜索深度: {self.max_depth}")
        print(f"运行时间: {self.runtime:.4f} 秒")
        print(f"启发函数类型: {self.heuristic_type}")

        if self.solution_moves:
            move_names = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
            arrow_moves = [move_names.get(move, move) for move in self.solution_moves]
            print(f"移动序列: {' → '.join(self.solution_moves)}")
            print(f"箭头表示: {' '.join(arrow_moves)}")
        print("=" * 60)

    def print_state(self, state: List[List[int]], title=""):
        """打印状态"""
        if title:
            print(f"\n{title}:")
        print("+" + "-" * 25 + "+")
        for i in range(4):
            print("|", end=" ")
            for j in range(4):
                if state[i][j] == 0:
                    print(f"{' ':4s}", end=" ")
                else:
                    print(f"{state[i][j]:4d}", end=" ")
            print("|")
        print("+" + "-" * 25 + "+")

    def get_move_names(self):
        """将移动序列转换为可读的名称"""
        move_names = {
            'U': '上移(↑)',
            'D': '下移(↓)',
            'L': '左移(←)',
            'R': '右移(→)'
        }
        return [move_names.get(move, move) for move in self.solution_moves]


# ==================== 测试实例生成 ====================

def create_test_cases():
    """创建测试实例"""
    test_cases = {}

    # 简单实例（3步解决）
    test_cases['简单'] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 0, 15]
    ]

    # 中等难度实例
    test_cases['中等'] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 0, 11],
        [13, 14, 15, 12]
    ]

    # 较难实例（需要更多步骤）
    test_cases['困难'] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 0, 10, 12],
        [13, 14, 11, 15]
    ]

    # 挑战性实例
    test_cases['挑战'] = [
        [0, 2, 3, 4],
        [1, 6, 7, 8],
        [5, 10, 11, 12],
        [9, 13, 14, 15]
    ]

    # 无解实例（不可解）
    test_cases['无解'] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 15, 14, 0]
    ]

    return test_cases


# ==================== 可视化功能 ====================

try:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from IPython.display import display, clear_output

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("注意：matplotlib未安装，可视化功能将不可用")
    print("请运行: pip install matplotlib")


def visualize_puzzle(state, title="", ax=None, show=True):
    """可视化十五数码状态"""
    if not HAS_MATPLOTLIB:
        print("可视化功能需要matplotlib库")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # 清除之前的图形
    ax.clear()

    # 设置坐标轴
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(True, linestyle='-', linewidth=2, color='black')
    ax.set_aspect('equal')

    # 隐藏坐标轴刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 绘制每个格子
    for i in range(4):
        for j in range(4):
            value = state[i][j]
            # 计算格子的中心位置
            x_center = j + 0.5
            y_center = 3.5 - i  # 反转y轴方向

            # 绘制格子背景
            if value == 0:
                # 空白格
                rect = patches.Rectangle((j, 3 - i), 1, 1,
                                         facecolor='lightgray',
                                         edgecolor='black',
                                         linewidth=2)
                ax.add_patch(rect)
                # 添加"空白"文本
                ax.text(x_center, y_center, "空白",
                        ha='center', va='center',
                        fontsize=12, fontweight='bold')
            else:
                # 数字格
                rect = patches.Rectangle((j, 3 - i), 1, 1,
                                         facecolor='lightblue',
                                         edgecolor='black',
                                         linewidth=2)
                ax.add_patch(rect)
                # 添加数字文本
                ax.text(x_center, y_center, str(value),
                        ha='center', va='center',
                        fontsize=14, fontweight='bold')

    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    if show:
        plt.show()

    return ax


def visualize_solution_interactive(solver, delay=0.5):
    """交互式可视化解决方案"""
    if not solver.solution_path:
        print("无解决方案可展示")
        return

    if not HAS_MATPLOTLIB:
        print("无法进行交互式可视化，matplotlib未安装")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    for step, state in enumerate(solver.solution_path):
        # 清除之前的输出
        clear_output(wait=True)

        if step == 0:
            title = "初始状态"
        else:
            move = solver.solution_moves[step - 1]
            move_name = {'U': '上移', 'D': '下移', 'L': '左移', 'R': '右移'}.get(move, move)
            title = f"第 {step} 步: {move_name} ({move})"

        # 绘制当前状态
        ax = visualize_puzzle(state, title, ax, show=False)

        # 显示当前步数和总步数
        plt.figtext(0.5, 0.01, f"步骤 {step}/{len(solver.solution_path) - 1}",
                    ha='center', fontsize=12, fontweight='bold')

        display(fig)

        # 如果不是最后一步，等待一下
        if step < len(solver.solution_path) - 1:
            time.sleep(delay)

    # 最后一步保持显示
    plt.show()


def print_solution_steps(solver):
    """打印解决方案的详细步骤"""
    if not solver.solution_path:
        print("无解决方案可展示")
        return

    print("\n" + "=" * 60)
    print("详细解决方案步骤")
    print("=" * 60)

    for step, state in enumerate(solver.solution_path):
        if step == 0:
            solver.print_state(state, "初始状态")
        else:
            move = solver.solution_moves[step - 1]
            move_name = {'U': '上移', 'D': '下移', 'L': '左移', 'R': '右移'}.get(move, move)
            solver.print_state(state, f"第 {step} 步: {move_name} ({move})")


# ==================== 性能分析功能 ====================

def compare_heuristics():
    """比较不同启发函数的性能"""
    test_cases = create_test_cases()

    print("=" * 80)
    print("不同启发函数性能比较")
    print("=" * 80)

    results = []

    for name, state in test_cases.items():
        if name == '无解':
            continue

        print(f"\n测试实例: {name}")

        for heuristic in ['manhattan', 'misplaced', 'euclidean']:
            solver = FifteenPuzzleSolver(state, heuristic_type=heuristic)

            if solver.is_solvable(state):
                start_time = time.time()
                success = solver.solve()
                end_time = time.time()

                if success:
                    print(
                        f"  {heuristic:<12}: 步数={len(solver.solution_moves):<4} 扩展节点={solver.nodes_expanded:<8} 时间={end_time - start_time:.4f}秒")

                    results.append({
                        'test': name,
                        'heuristic': heuristic,
                        'time': end_time - start_time,
                        'expanded': solver.nodes_expanded,
                        'steps': len(solver.solution_moves)
                    })

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("性能比较汇总")
    print("=" * 80)
    print(f"{'测试实例':<10} {'启发函数':<12} {'步数':<8} {'扩展节点':<12} {'时间(秒)':<10}")
    print("-" * 80)

    for result in results:
        print(f"{result['test']:<10} {result['heuristic']:<12} {result['steps']:<8} "
              f"{result['expanded']:<12} {result['time']:<10.4f}")

    return results


# ==================== 演示函数 ====================

def demo_fifteen_puzzle():
    """演示十五数码问题的求解过程"""
    print("=" * 60)
    print("十五数码问题求解演示")
    print("=" * 60)

    # 创建测试实例
    test_cases = create_test_cases()

    # 选择中等难度实例
    initial_state = test_cases['中等']

    print("初始状态:")
    solver = FifteenPuzzleSolver(initial_state, heuristic_type='manhattan')
    solver.print_state(initial_state)

    # 检查可解性
    is_solvable = solver.is_solvable(initial_state)
    print(f"\n可解性检查: {'可解' if is_solvable else '不可解'}")

    if is_solvable:
        # 求解
        print("\n正在使用A*算法求解...")
        start_time = time.time()
        success = solver.solve()
        end_time = time.time()

        if success:
            print(f"求解成功! 耗时: {end_time - start_time:.4f}秒")

            # 显示统计信息
            solver.print_statistics()

            # 询问是否可视化
            if HAS_MATPLOTLIB:
                print("\n是否可视化解决方案? (输入 'y' 确认，其他跳过): ")
                visualize = input().strip().lower()

                if visualize == 'y':
                    print("开始可视化解决方案...")
                    visualize_solution_interactive(solver, delay=0.8)
            else:
                print("\n详细步骤:")
                print_solution_steps(solver)
        else:
            print("求解失败!")
    else:
        print("该初始状态无解!")


def interactive_solver():
    """交互式求解器"""
    print("=" * 60)
    print("交互式十五数码求解器")
    print("=" * 60)

    # 创建测试实例
    test_cases = create_test_cases()

    # 显示可用的测试实例
    print("可用的测试实例:")
    for i, (name, state) in enumerate(test_cases.items(), 1):
        print(f"{i}. {name}")

    print(f"{len(test_cases) + 1}. 自定义输入")

    try:
        choice = int(input("\n请选择测试实例 (输入序号): "))

        if 1 <= choice <= len(test_cases):
            test_names = list(test_cases.keys())
            selected_test = test_names[choice - 1]
            initial_state = test_cases[selected_test]
        elif choice == len(test_cases) + 1:
            # 自定义输入
            print("\n自定义输入模式")
            print("请输入4x4的十五数码状态（用空格分隔数字，0表示空白）")
            print("示例: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0")

            input_str = input("\n请输入16个数字: ")
            numbers = list(map(int, input_str.strip().split()))

            if len(numbers) != 16:
                print("错误: 需要16个数字!")
                return

            # 转换为4x4矩阵
            initial_state = [numbers[i:i + 4] for i in range(0, 16, 4)]
            selected_test = "自定义"
        else:
            print("无效选择，使用默认实例")
            initial_state = test_cases['中等']
            selected_test = '中等'
    except:
        print("无效输入，使用默认实例")
        initial_state = test_cases['中等']
        selected_test = '中等'

    # 选择启发函数
    print("\n选择启发函数:")
    print("1. 曼哈顿距离 (推荐)")
    print("2. 不在位元素个数")
    print("3. 欧几里得距离")

    try:
        heuristic_choice = int(input("请选择启发函数 (输入序号): "))
        if heuristic_choice == 1:
            heuristic_type = 'manhattan'
        elif heuristic_choice == 2:
            heuristic_type = 'misplaced'
        elif heuristic_choice == 3:
            heuristic_type = 'euclidean'
        else:
            print("无效选择，使用曼哈顿距离")
            heuristic_type = 'manhattan'
    except:
        print("无效输入，使用曼哈顿距离")
        heuristic_type = 'manhattan'

    # 创建求解器
    print(f"\n求解实例: {selected_test}")
    print(f"使用启发函数: {heuristic_type}")

    solver = FifteenPuzzleSolver(initial_state, heuristic_type=heuristic_type)

    # 显示初始状态
    print("\n初始状态:")
    solver.print_state(initial_state)

    # 检查可解性
    is_solvable = solver.is_solvable(initial_state)
    print(f"\n可解性检查: {'可解' if is_solvable else '不可解'}")

    if not is_solvable:
        print("该初始状态无解!")
        return

    # 求解
    print("\n正在使用A*算法求解...")
    start_time = time.time()
    success = solver.solve()
    end_time = time.time()

    if success:
        print(f"求解成功! 耗时: {end_time - start_time:.4f}秒")

        # 显示统计信息
        solver.print_statistics()

        # 询问是否可视化
        if HAS_MATPLOTLIB:
            print("\n是否可视化解决方案? (输入 'y' 确认，其他跳过): ")
            visualize = input().strip().lower()

            if visualize == 'y':
                print("开始可视化解决方案...")
                visualize_solution_interactive(solver, delay=0.8)

        # 询问是否显示详细步骤
        print("\n是否显示详细步骤? (输入 'y' 确认，其他跳过): ")
        show_details = input().strip().lower()

        if show_details == 'y':
            print_solution_steps(solver)
    else:
        print("求解失败!")


def benchmark_all_cases():
    """对所有可解测试实例进行性能测试"""
    test_cases = create_test_cases()

    print("性能测试结果")
    print("=" * 80)
    print(f"{'测试实例':<15} {'启发函数':<12} {'步数':<6} {'扩展节点':<10} {'生成节点':<10} {'时间(秒)':<10}")
    print("-" * 80)

    results = []

    for name, state in test_cases.items():
        if name == '无解':
            continue

        for heuristic in ['manhattan', 'misplaced', 'euclidean']:
            solver = FifteenPuzzleSolver(state, heuristic_type=heuristic)

            if solver.is_solvable(state):
                start_time = time.time()
                success = solver.solve()
                end_time = time.time()

                if success:
                    results.append({
                        'test': name,
                        'heuristic': heuristic,
                        'steps': len(solver.solution_moves),
                        'expanded': solver.nodes_expanded,
                        'generated': solver.nodes_generated,
                        'time': end_time - start_time
                    })

                    print(f"{name:<15} {heuristic:<12} {len(solver.solution_moves):<6} "
                          f"{solver.nodes_expanded:<10} {solver.nodes_generated:<10} "
                          f"{end_time - start_time:<10.4f}")

    print("=" * 80)
    return results


def analyze_algorithm_performance():
    """分析算法性能"""
    print("=" * 60)
    print("A*算法性能深度分析")
    print("=" * 60)

    test_cases = create_test_cases()

    # 只分析可解实例
    solvable_cases = {k: v for k, v in test_cases.items() if k != '无解'}

    all_results = []

    for test_name, state in solvable_cases.items():
        print(f"分析实例: {test_name}")

        for heuristic in ['manhattan', 'misplaced', 'euclidean']:
            solver = FifteenPuzzleSolver(state, heuristic_type=heuristic)

            if solver.is_solvable(state):
                success = solver.solve()

                if success:
                    all_results.append({
                        'test': test_name,
                        'heuristic': heuristic,
                        'time': solver.runtime,
                        'expanded': solver.nodes_expanded,
                        'generated': solver.nodes_generated,
                        'steps': len(solver.solution_moves),
                        'depth': solver.max_depth
                    })

    # 打印详细数据
    print("\n详细性能数据:")
    print("=" * 80)
    print(
        f"{'测试实例':<10} {'启发函数':<12} {'时间(秒)':<10} {'扩展节点':<12} {'生成节点':<12} {'步数':<8} {'深度':<8}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['test']:<10} {result['heuristic']:<12} {result['time']:<10.4f} "
              f"{result['expanded']:<12} {result['generated']:<12} {result['steps']:<8} {result['depth']:<8}")

    return all_results


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("十五数码问题求解系统")
    print("=" * 60)
    print("1. 基本演示")
    print("2. 比较不同启发函数")
    print("3. 性能基准测试")
    print("4. 交互式求解器")
    print("5. 算法性能分析")
    print("6. 退出")

    while True:
        try:
            choice = input("\n请选择功能 (1-6): ").strip()

            if choice == '1':
                demo_fifteen_puzzle()
            elif choice == '2':
                compare_heuristics()
            elif choice == '3':
                benchmark_all_cases()
            elif choice == '4':
                interactive_solver()
            elif choice == '5':
                analyze_algorithm_performance()
            elif choice == '6':
                print("感谢使用十五数码问题求解系统，再见!")
                break
            else:
                print("无效选择，请重新输入")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")


def run_single_example():
    """运行单个示例"""
    print("\n运行单个示例...")

    # 使用中等难度实例
    test_cases = create_test_cases()
    initial_state = test_cases['中等']

    print("初始状态:")
    solver = FifteenPuzzleSolver(initial_state, heuristic_type='manhattan')
    solver.print_state(initial_state)

    print("\n检查可解性...")
    if not solver.is_solvable(initial_state):
        print("该初始状态无解!")
        return

    print("可解!")
    print("\n使用A*算法求解...")

    start_time = time.time()
    success = solver.solve()
    end_time = time.time()

    if success:
        print(f"求解成功! 耗时: {end_time - start_time:.4f}秒")
        solver.print_statistics()

        # 显示前几步
        print("\n解决方案前5步:")
        for i in range(min(5, len(solver.solution_path))):
            if i == 0:
                solver.print_state(solver.solution_path[i], "初始状态")
            else:
                move = solver.solution_moves[i - 1]
                move_name = {'U': '上移', 'D': '下移', 'L': '左移', 'R': '右移'}.get(move, move)
                solver.print_state(solver.solution_path[i], f"第{i}步: {move_name}")
    else:
        print("求解失败!")


# ==================== 直接运行时的处理 ====================

if __name__ == "__main__":
    # 检查是否需要安装依赖
    if not HAS_MATPLOTLIB:
        print("注意: matplotlib未安装，可视化功能将不可用")
        print("安装命令: pip install matplotlib")
        print("继续运行...\n")

    # 如果提供了命令行参数，则运行单个示例
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        run_single_example()
    else:
        main()