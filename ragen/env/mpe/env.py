import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import re
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from gym import spaces

INSTRUCTION_TEMPLATE = """You are agent {agent_id} in a multi-agent navigation system.

Multi-Goal Navigation Task Guide:
Objective: All agents need to collaborate, with each agent reaching a goal point.

Symbol Legend:
. Empty space | # Obstacle | O Goal point | A Yourself | B Other agents
Å You on a goal point | Ḃ Other agent on a goal point

Action Choices:
<answer>Stay</answer> | <answer>Left</answer> | <answer>Right</answer> | <answer>Down</answer> | <answer>Up</answer>

Reward Mechanism:
Each move: {step_penalty}
Reaching a goal point: {goal_reward}
All goal points occupied by agents: {completion_reward}

[Current Observation]:
{observation}

Please decide your next action. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""



INSTRUCTION_TEMPLATE = """You are agent {agent_id} in a multi-agent navigation system.

Multi-Goal Navigation Task Guide:
Objective: All agents need to collaborate, with each agent reaching a goal point, while avoiding collisions with other agents and obstacles.

Symbol Legend:
. Empty space | # Obstacle | O Goal point | A Yourself | B Other agents
Å You on a goal point | Ḃ Other agent on a goal point

#### Actions
Each agent can perform one of these actions:
- **Stay (0)**: Remain in the current position
- **Left (1)**: Move one cell to the left
- **Right (2)**: Move one cell to the right
- **Down (3)**: Move one cell down
- **Up (4)**: Move one cell up
---

### Reward Mechanism:
- Each move: `{step_penalty}` (penalty applied per step to encourage efficient movement)  
- Collision penalty: `{collision_penalty}` (penalty if an agent tries to occupy the same space as another agent or obstacle)  
- Reaching a goal point: `{goal_reward}` (reward for successfully reaching a goal)  
- All goal points occupied by agents: `{completion_reward}` (bonus reward when all agents reach their goals)  
- Distance penalty per step: `{distance_factor}` (negative reward proportional to the remaining distance to the goal, encouraging shorter paths)  

[Current Observation]:
{observation}

Please decide your next action. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 模型提示模板
templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output format: <reasoning> [Your reasoning process] </reasoning> <answer> [Your answer] </answer>, with no extra text. Strictly follow this format.<|im_end|>\n<|im_start|>assistant\n<reasoning>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <reasoning> </reasoning> tags. And return the final answer in <answer> </answer> tags, for example <reasoning> [Reasoning process] </reasoning> <answer> Left </answer>\nAssistant: \n<reasoning>'
}

class MultiAgentGridEnv:
    """
    基于网格的多智能体环境的基类
    """
    def __init__(self, grid_size: int = 10, max_steps: int = 100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0
        
        # 定义动作空间
        self.ACTION_SPACE = spaces.Discrete(5)  # 5个动作：不动，上下左右
        
    def reset(self, seed: Optional[int] = None, mode: str = 'tiny_rgb_array'):
        """重置环境"""
        self.step_count = 0
        
    def step(self, action: int):
        """执行一步"""
        self.step_count += 1
        
    def render(self, mode='human'):
        """渲染环境"""
        pass
        
    def close(self):
        """关闭环境"""
        pass
        
    def success(self) -> bool:
        """检查是否成功完成任务"""
        return False
        
    def finished(self) -> bool:
        """检查环境是否结束"""
        return self.success() or self.step_count >= self.max_steps
        
    def copy(self):
        """创建环境的深拷贝"""
        pass
        
    def extract_action(self, text: str) -> int:
        """从文本中提取动作"""
        pass

class MultiGoalNavEnv(MultiAgentGridEnv):
    """
    多目标导航的多智能体网格环境
    
    ## 描述
    这个环境中有N个智能体和N个目标点，所有智能体共享这些目标点，需要协作完成导航任务。
    智能体根据与目标点的距离获得奖励，并在与其他智能体碰撞时受到惩罚。
    
    ## 动作空间
    动作空间是离散的，有5个动作：
    - 0: 不移动
    - 1: 向左移动
    - 2: 向右移动
    - 3: 向下移动
    - 4: 向上移动
    
    ## 观测空间
    每个智能体接收一个局部的网格观测，包含：
    - 自己的位置
    - 其他智能体的位置
    - 目标点的位置
    
    ## 奖励
    - 与最近目标点距离越近，奖励越高（负距离）
    - 与其他智能体碰撞时受到惩罚
    - 到达目标点时获得额外奖励
    - 所有目标点都有智能体时获得任务完成奖励
    
    ## 碰撞规则
    - 智能体的碰撞体积为自身位置
    - 当两个智能体位于同一位置时，视为发生碰撞
    """
    
    # 网格单元类型
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2
    GOAL = 3
    AGENT_ON_GOAL = 4
    
    # 观测中的智能体类型
    SELF = 5  # 自己
    OTHER_AGENT = 6  # 其他智能体
    SELF_ON_GOAL = 7  # 自己在目标点上
    OTHER_ON_GOAL = 8  # 其他智能体在目标点上
    
    # 动作定义
    STAY = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    UP = 4
    
    # 动作映射到坐标变化
    ACTION_DELTA = {
        STAY: (0, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
        DOWN: (1, 0),
        UP: (-1, 0)
    }
    
    ACTION_LOOKUP = {
        STAY: "Stay",
        LEFT: "Left",
        RIGHT: "Right",
        DOWN: "Down",
        UP: "Up"
    }
    
    # 奖励参数
    STEP_PENALTY = -0.1  # 每步的惩罚
    COLLISION_PENALTY = -1.0  # 碰撞惩罚
    GOAL_REWARD = 0  # 到达目标点的奖励
    COMPLETION_REWARD = 10.0  # 完成所有目标点的奖励
    DISTANCE_FACTOR = -0.1  # 距离因子（负值，距离越近奖励越高）
    
    def __init__(
        self, 
        grid_size: int = 10, 
        num_agents: int = 3, 
        num_obstacles: int = 0, 
        max_steps: int = 100,
        obstacle_positions: Optional[List[Tuple[int, int]]] = None,
        agent_positions: Optional[List[Tuple[int, int]]] = None,
        goal_positions: Optional[List[Tuple[int, int]]] = None,
        seed: Optional[int] = None
    ):
        """
        初始化多目标导航环境
        
        Args:
            grid_size: 网格大小
            num_agents: 智能体数量
            num_obstacles: 障碍物数量（默认为0，不生成障碍物）
            max_steps: 最大步数
            obstacle_positions: 障碍物位置，如果为None则随机生成
            agent_positions: 智能体初始位置，如果为None则随机生成
            goal_positions: 目标点位置，如果为None则随机生成
            seed: 随机种子
            
        注意:
            - 智能体的碰撞体积为自身周围一个单位的格子（曼哈顿距离<=1）
            - 生成的目标点之间至少相隔一个网格
            - 智能体之间也至少相隔一个网格
            - 障碍物、智能体和目标点之间也至少相隔一个网格
        """
        super().__init__(grid_size, max_steps)
        
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化网格
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.seed = seed
        
        # 初始化智能体、目标点和障碍物的位置
        self.obstacle_positions = obstacle_positions
        self.initial_agent_positions = agent_positions
        self.goal_positions = goal_positions
        
        # 重置环境
        self.reset()
    
    def _generate_random_positions(self, num_positions: int, occupied_positions: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        生成不与已占用位置重叠的随机位置
        
        Args:
            num_positions: 需要生成的位置数量
            occupied_positions: 已占用的位置集合
            
        Returns:
            List[Tuple[int, int]]: 生成的随机位置列表
        """
        positions = []
        while len(positions) < num_positions:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if pos not in occupied_positions:
                positions.append(pos)
                occupied_positions.add(pos)
        return positions
    
    def reset(self, seed: Optional[int] = None, mode: str = 'tiny_rgb_array') -> Dict[str, str]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            mode: 渲染模式
                
        Returns:
            Dict[str, str]: 每个智能体的观测（text格式）
        """
        # 重置步数
        self.step_count = 0
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 清空网格
        self.grid.fill(self.EMPTY)
        
        # 已占用的位置集合
        occupied_positions = set()
        
        # 放置障碍物
        if self.obstacle_positions is None:
            self.obstacle_positions = self._generate_random_positions(self.num_obstacles, occupied_positions)
        else:
            occupied_positions.update(self.obstacle_positions)
            
        for pos in self.obstacle_positions:
            self.grid[pos] = self.OBSTACLE
        
        # 放置智能体
        if self.initial_agent_positions is None:
            self.agent_positions = self._generate_random_positions(self.num_agents, occupied_positions)
        else:
            self.agent_positions = self.initial_agent_positions.copy()
            occupied_positions.update(self.agent_positions)
            
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        
        for pos in self.agent_positions:
            self.grid[pos] = self.AGENT
        
        # 放置目标点
        if self.goal_positions is None:
            self.goal_positions = self._generate_random_positions(self.num_agents, occupied_positions)
        else:
            occupied_positions.update(self.goal_positions)
            
        for pos in self.goal_positions:
            self.grid[pos] = self.GOAL
        
        # 初始化目标点状态
        self.agents_on_goals = [None] * len(self.goal_positions)  # 记录每个目标点上的智能体
        
        # 返回每个智能体的观测
        return self.render_observations(mode=mode)
    
    def step(self, action: Union[int, List[int]]) -> Tuple[Dict[str, str], Union[float, List[float]], bool, Dict[str, Any]]:
        """
        执行一步
        
        Args:
            action: 动作编号或动作列表（联合动作）
            
        Returns:
            Tuple[Dict[str, str], Union[float, List[float]], bool, Dict[str, Any]]:
                - 每个智能体的观测（text格式）
                - 奖励（单个值或列表）
                - 是否结束
                - 额外信息
        """
        # 检查是否已经结束
        if self.finished():
            return self.render_observations(mode='text'), [0.0] * self.num_agents, True, {"action_is_effective": [False] * self.num_agents}
        
        # 增加步数
        self.step_count += 1
        print(action)
        # 处理单个动作的情况（向后兼容）
        if isinstance(action, int):
            action = [action] + [self.STAY] * (self.num_agents - 1)
        
        # 确保动作列表长度与智能体数量一致
        if len(action) != self.num_agents:
            raise ValueError(f"动作列表长度 ({len(action)}) 与智能体数量 ({self.num_agents}) 不匹配")
        
        # 记录之前的位置
        prev_agent_positions = self.agent_positions.copy()
        
        # 记录动作是否有效
        action_is_effective = [False] * self.num_agents
        print(action)
        # 计算每个智能体的新位置
        new_positions = []
        for i, a in enumerate(action):
            agent_pos = self.agent_positions[i]
            print(a)
            # 如果是无效动作，保持原位
            if a == self.STAY:
                new_positions.append(agent_pos)
                continue
            
            # 获取动作对应的位置变化
            delta = self.ACTION_DELTA[a]
            
            # 计算新位置
            new_pos = (agent_pos[0] + delta[0], agent_pos[1] + delta[1])
            
            # 检查新位置是否有效
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size and 
                self.grid[new_pos] != self.OBSTACLE):
                
                # 检查是否与其他智能体的新位置冲突
                if new_pos not in new_positions:
                    new_positions.append(new_pos)
                    action_is_effective[i] = True
                else:
                    # 冲突，保持原位
                    new_positions.append(agent_pos)
            else:
                # 无效位置，保持原位
                new_positions.append(agent_pos)
        
        # 更新网格和智能体位置
        # 首先清除所有智能体
        for pos in self.agent_positions:
            if self.grid[pos] == self.AGENT_ON_GOAL:
                self.grid[pos] = self.GOAL
            elif self.grid[pos] == self.AGENT:
                self.grid[pos] = self.EMPTY
        
        # 更新智能体位置
        self.agent_positions = new_positions
        
        # 重新放置智能体
        self.agents_on_goals = [None] * len(self.goal_positions)
        for i, pos in enumerate(self.agent_positions):
            # 检查是否在目标点上
            on_goal = False
            for j, goal_pos in enumerate(self.goal_positions):
                if pos == goal_pos:
                    self.grid[pos] = self.AGENT_ON_GOAL
                    self.agents_on_goals[j] = self.agents[i]
                    on_goal = True
                    break
            
            if not on_goal:
                self.grid[pos] = self.AGENT
        
        # 计算共享奖励
        shared_reward = self.STEP_PENALTY  # 每步的基础惩罚
        
        # 检查碰撞情况
        collision_count = 0
        position_counts = {}
        for pos in self.agent_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        for pos, count in position_counts.items():
            if count > 1:
                collision_count += count
        
        # 添加碰撞惩罚
        if collision_count > 0:
            shared_reward += self.COLLISION_PENALTY * collision_count
        
        # 检查智能体是否在目标点上
        agents_on_goal_count = 0
        for i, agent_pos in enumerate(self.agent_positions):
            for goal_pos in self.goal_positions:
                if agent_pos == goal_pos:
                    agents_on_goal_count += 1
                    shared_reward += self.GOAL_REWARD
                    break
        
        # 检查是否所有目标点都有智能体
        done = self.success() or self.step_count >= self.max_steps
        
        # 如果完成任务，给予额外奖励
        if self.success():
            shared_reward += self.COMPLETION_REWARD
        
        # 所有智能体共享相同的奖励
        rewards = [shared_reward] * self.num_agents
        
        # 返回观测、奖励、是否结束和额外信息
        return self.render_observations(mode='text'), rewards, done, {
            "action_is_effective": action_is_effective,
            "collision_count": collision_count,
            "agents_on_goal_count": agents_on_goal_count
        }
    
    def get_observations(self) -> Dict[str, List[List[int]]]:
        """
        获取每个智能体的观测
        
        Returns:
            Dict[str, List[List[int]]]: 每个智能体的观测
        """
        observations = {}
        
        for i, agent in enumerate(self.agents):
            # 创建观测网格（与环境网格相同）
            observation = self.grid.copy()
            
            # 标记自己和其他智能体的位置
            for j, pos in enumerate(self.agent_positions):
                if i == j:  # 自己
                    # 检查是否在目标点上
                    is_on_goal = False
                    for goal_pos in self.goal_positions:
                        if pos == goal_pos:
                            is_on_goal = True
                            observation[pos] = self.SELF_ON_GOAL
                            break
                    
                    if not is_on_goal:
                        observation[pos] = self.SELF
                else:  # 其他智能体
                    # 检查是否在目标点上
                    is_on_goal = False
                    for goal_pos in self.goal_positions:
                        if pos == goal_pos:
                            is_on_goal = True
                            observation[pos] = self.OTHER_ON_GOAL
                            break
                    
                    if not is_on_goal:
                        observation[pos] = self.OTHER_AGENT
            
            observations[agent] = observation
        
        return observations
    
    def render_observations(self, mode: str = 'text') -> Dict[str, str]:
        """
        将每个智能体的观测渲染为指定格式
        
        Args:
            mode: 渲染模式，可以是'text', 'tiny_rgb_array', 'list'
            
        Returns:
            Dict[str, str]: 每个智能体的渲染后的观测
        """
        # 获取每个智能体的观测
        observations = self.get_observations()
        rendered_observations = {}
        
        for agent_id, observation in observations.items():
            if mode == 'text':
                # 创建文本表示
                symbols = {
                    self.EMPTY: '.',         # 空白
                    self.OBSTACLE: '#',      # 障碍物
                    self.AGENT: 'A',         # 智能体（不应出现在观测中）
                    self.GOAL: 'O',          # 目标点
                    self.AGENT_ON_GOAL: 'X', # 智能体在目标点上（不应出现在观测中）
                    self.SELF: 'A',          # 自己
                    self.OTHER_AGENT: 'B',   # 其他智能体
                    self.SELF_ON_GOAL: 'Å',  # 自己在目标点上
                    self.OTHER_ON_GOAL: 'Ḃ',  # 其他智能体在目标点上
                }
                
                # 转换为文本
                text_grid = []
                for row in observation:
                    text_row = ' '.join([symbols.get(cell, '?') for cell in row])
                    text_grid.append(text_row)
                
                rendered_observations[agent_id] = '\n'.join(text_grid)
                
            elif mode == 'tiny_rgb_array':
                # 创建文本表示，但以字符串形式返回
                symbols = {
                    self.EMPTY: '.',         # 空白
                    self.OBSTACLE: '#',      # 障碍物
                    self.AGENT: 'A',         # 智能体（不应出现在观测中）
                    self.GOAL: 'O',          # 目标点
                    self.AGENT_ON_GOAL: 'X', # 智能体在目标点上（不应出现在观测中）
                    self.SELF: 'A',          # 自己
                    self.OTHER_AGENT: 'B',   # 其他智能体
                    self.SELF_ON_GOAL: 'Å',  # 自己在目标点上
                    self.OTHER_ON_GOAL: 'Ḃ',  # 其他智能体在目标点上
                }
                
                # 转换为文本并连接
                text_rows = []
                for row in observation:
                    text_row = ''.join([symbols.get(cell, '?') for cell in row])
                    text_rows.append(text_row)
                
                rendered_observations[agent_id] = '\n'.join(text_rows)
                
            elif mode == 'list':
                # 类似于text模式，但每个单元格都有更多空格
                symbols = {
                    self.EMPTY: ' . ',       # 空白
                    self.OBSTACLE: ' # ',    # 障碍物
                    self.AGENT: ' A ',       # 智能体（不应出现在观测中）
                    self.GOAL: ' O ',        # 目标点
                    self.AGENT_ON_GOAL: ' X ', # 智能体在目标点上（不应出现在观测中）
                    self.SELF: ' A ',        # 自己
                    self.OTHER_AGENT: ' B ', # 其他智能体
                    self.SELF_ON_GOAL: ' Å ', # 自己在目标点上
                    self.OTHER_ON_GOAL: ' Ḃ ', # 其他智能体在目标点上
                }
                
                # 转换为文本列表
                text_grid = []
                for row in observation:
                    text_row = ' '.join([symbols.get(cell, ' ? ') for cell in row])
                    text_grid.append(text_row)
                
                rendered_observations[agent_id] = text_grid
            
            else:
                raise ValueError(f"不支持的渲染模式: {mode}")
        
        return rendered_observations
    
    def render(self, mode: str = 'human') -> Union[np.ndarray, List[str], None]:
        """
        渲染环境
        
        Args:
            mode: 渲染模式，可以是'human', 'rgb_array', 'text', 'tiny_rgb_array', 'list', 'state'
            
        Returns:
            Union[np.ndarray, List[str], None]: 根据模式返回不同类型的渲染结果
        """
        if mode == 'human':
            plt.figure(figsize=(8, 8))
            plt.imshow(self.get_image(mode='rgb_array'))
            plt.axis('off')
            plt.show()
            return None
            
        elif mode == 'rgb_array':
            # 创建彩色网格
            colors = {
                self.EMPTY: [1.0, 1.0, 1.0],  # 白色
                self.OBSTACLE: [0.5, 0.5, 0.5],  # 灰色
                self.AGENT: [0.0, 0.0, 1.0],  # 蓝色
                self.GOAL: [0.0, 1.0, 0.0],  # 绿色
                self.AGENT_ON_GOAL: [1.0, 0.0, 1.0],  # 紫色
            }
            
            # 创建RGB图像
            rgb_grid = np.zeros((self.grid_size, self.grid_size, 3))
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_type = self.grid[i, j]
                    rgb_grid[i, j] = colors.get(cell_type, [0, 0, 0])
            
            return rgb_grid
            
        elif mode == 'text':
            # 创建文本表示
            symbols = {
                self.EMPTY: '.',         # 空白
                self.OBSTACLE: '#',      # 障碍物
                self.AGENT: 'A',         # 智能体
                self.GOAL: 'O',          # 目标点
                self.AGENT_ON_GOAL: 'X', # 智能体在目标点上
                self.SELF: 'A',          # SELF（在观测中使用）
                self.OTHER_AGENT: 'B',   # OTHER_AGENT（在观测中使用）
                self.SELF_ON_GOAL: 'Å',  # SELF_ON_GOAL（在观测中使用）
                self.OTHER_ON_GOAL: 'Ḃ',  # OTHER_ON_GOAL（在观测中使用）
            }
            
            # 创建网格副本
            grid_copy = self.grid.copy()
            
            # 转换为文本
            text_grid = []
            for row in grid_copy:
                text_row = ' '.join([symbols[cell] for cell in row])
                text_grid.append(text_row)
            
            return text_grid
        
        elif mode == 'tiny_rgb_array':
            # 创建文本表示，但以字符串形式返回
            symbols = {
                self.EMPTY: '.',         # 空白
                self.OBSTACLE: '#',      # 障碍物
                self.AGENT: 'A',         # 智能体
                self.GOAL: 'O',          # 目标点
                self.AGENT_ON_GOAL: 'X', # 智能体在目标点上
            }
            
            # 创建网格副本
            grid_copy = self.grid.copy()
            
            # 转换为文本并连接
            text_rows = []
            for row in grid_copy:
                text_row = ''.join([symbols[cell] for cell in row])
                text_rows.append(text_row)
            
            return '\n'.join(text_rows)
            
        elif mode == 'list':
            # 类似于text模式，但每个单元格都有更多空格
            symbols = {
                self.EMPTY: ' . ',       # 空白
                self.OBSTACLE: ' # ',    # 障碍物
                self.AGENT: ' A ',       # 智能体
                self.GOAL: ' O ',        # 目标点
                self.AGENT_ON_GOAL: ' X ', # 智能体在目标点上
            }
            
            # 创建网格副本
            grid_copy = self.grid.copy()
            
            # 转换为文本列表
            text_grid = []
            for row in grid_copy:
                text_row = ' '.join([symbols[cell] for cell in row])
                text_grid.append(text_row)
            
            return text_grid
            
        elif mode == 'state':
            # 返回当前网格状态
            return self.grid.copy()
        
        else:
            raise ValueError(f"不支持的渲染模式: {mode}")
    
    def get_image(self, mode: str = 'rgb_array') -> np.ndarray:
        """
        获取环境的图像表示
        
        Args:
            mode: 渲染模式
            
        Returns:
            np.ndarray: 环境的图像表示
        """
        if mode == 'rgb_array':
            return self.render(mode='rgb_array')
        else:
            raise ValueError(f"不支持的图像模式: {mode}")
    
    def copy(self):
        """
        创建环境的深拷贝
        
        Returns:
            MultiGoalNavEnv: 环境的深拷贝
        """
        new_self = MultiGoalNavEnv(
            grid_size=self.grid_size,
            num_agents=self.num_agents,
            num_obstacles=self.num_obstacles,
            max_steps=self.max_steps
        )
        new_self.grid = self.grid.copy()
        new_self.obstacle_positions = self.obstacle_positions.copy() if self.obstacle_positions else None
        new_self.agent_positions = self.agent_positions.copy() if self.agent_positions else None
        new_self.goal_positions = self.goal_positions.copy() if self.goal_positions else None
        new_self.agents = self.agents.copy()
        new_self.agents_on_goals = self.agents_on_goals.copy()
        new_self.step_count = self.step_count
        return new_self
    
    def extract_action(self, text: str) -> int:
        """
        从文本中提取动作
        
        Args:
            text: 文本
            
        Returns:
            int: 动作编号
        """
        # 检查是否包含<answer>标签
        if "<answer>" in text and "</answer>" in text:
            # 提取<answer>标签中的内容
            answer = text.split("<answer>")[1].split("</answer>")[0].strip()
            
            # 将动作映射到编号
            action_map = {
                "stay": self.STAY,
                "left": self.LEFT,
                "right": self.RIGHT,
                "down": self.DOWN,
                "up": self.UP
            }
            
            # 不区分大小写
            answer_lower = answer.lower()
            if answer_lower in action_map:
                return action_map[answer_lower]
        
        # 如果没有找到有效动作，返回不动
        return self.STAY
    
    def success(self) -> bool:
        """
        检查是否所有目标点都有智能体
        
        Returns:
            bool: 如果所有目标点都有智能体，则返回True
        """
        # 检查是否所有目标点都有智能体（None表示没有智能体）
        return all(agent is not None for agent in self.agents_on_goals)
    
    def finished(self) -> bool:
        """
        检查环境是否结束（成功完成或达到最大步数）
        
        Returns:
            bool: 如果环境结束，则返回True
        """
        return self.success() or self.step_count >= self.max_steps
    
    def close(self) -> None:
        """关闭环境"""
        plt.close('all')

GUIDE = """
### Multi-Agent Goal Navigation Instructions

In this environment, multiple agents need to navigate to goal positions in a shared grid world. This requires coordination and strategic planning to avoid collisions while reaching goals efficiently.

---

#### Symbols and Their Meaning
- **Empty Space (`.`)**: Open spaces where agents can move freely.
- **Obstacle (`#`)**: These block movement. Agents cannot move through obstacles.
- **Goal (`O`)**: Target positions that agents need to reach.
- **Self (`A`)**: Represents the observing agent's own position.
- **Other Agent (`B`)**: Represents the position of other agents.
- **Self on Goal (`Å`)**: Represents the observing agent positioned on a goal.
- **Other Agent on Goal (`Ḃ`)**: Represents other agents positioned on goals.

---

#### Actions
Each agent can perform one of these actions:
- **Stay (0)**: Remain in the current position
- **Left (1)**: Move one cell to the left
- **Right (2)**: Move one cell to the right
- **Down (3)**: Move one cell down
- **Up (4)**: Move one cell up

---

#### Your Goal
Guide all agents to goal positions while avoiding collisions with other agents and obstacles. The task is completed when all goals are covered by agents.

---

#### Rewards
- Agents receive higher rewards when closer to goal positions
- Collisions with other agents result in penalties
- Reaching a goal position grants a bonus reward
- When all goals are covered, a completion reward is given

---

#### Tips
- Plan movements carefully to avoid collisions
- Coordinate agent movements to efficiently cover all goals
- Sometimes it's better for an agent to wait than to move into a crowded area
- The optimal solution minimizes the total distance traveled by all agents
"""

if __name__ == '__main__':
    # 创建环境
    env = MultiGoalNavEnv(grid_size=10, num_agents=3, num_obstacles=0, max_steps=100, seed=42)
    
    # 重置环境
    observations = env.reset()
    
    # 保存初始图像
    init_img = env.get_image('rgb_array')
    plt.imsave('multi_goal_nav_init.png', init_img)
    
    # 保存文本表示
    with open('multi_goal_nav_init_text.txt', 'w', encoding='utf-8') as f:
        text_grid = env.render(mode='text')
        f.write('\n'.join(text_grid))
    
    print("初始状态已保存到multi_goal_nav_init.png和multi_goal_nav_init_text.txt")
    
    # 打印初始状态
    print("\n初始网格:")
    for line in env.render(mode='text'):
        print(line)
    
    # 打印智能体位置
    print("\n智能体位置:")
    for i, agent in enumerate(env.agents):
        print(f"{agent}: {env.agent_positions[i]}")
    
    # 打印目标点位置
    print("\n目标点位置:")
    for i, pos in enumerate(env.goal_positions):
        print(f"目标 {i+1}: {pos}")
    
    # 交互式控制环境
    done = False
    step_count = 0
    
    print("\n\n开始交互式控制环境")
    print("使用WASD键控制移动方向:")
    print("  W = 上")
    print("  A = 左")
    print("  S = 下")
    print("  D = 右")
    print("  0 = 不动")
    print("  q = 退出")
    
    while not done and step_count < env.max_steps:
        step_count += 1
        print(f"\n===== 步骤 {step_count} =====")
        
        # 获取动作
        while True:
            action_input = input(f"请输入动作 (W=上, A=左, S=下, D=右, 0=不动, q=退出): ")
            
            if action_input.lower() == 'q':
                print("用户选择退出")
                break
            
            action = env.extract_action(action_input)
            print(action)
            break
        
        # 检查是否退出
        if action_input.lower() == 'q':
            break

        print(action)
        # 执行动作
        observations, reward, done, info = env.step(action)
        # 显示结果
        print("\n执行动作结果:")
        print(f"动作: {env.ACTION_LOOKUP[action]}")

        # 显示奖励
        formatted_rewards = ", ".join(f"{r:.3f}" for r in reward)
        print(f"\n奖励: [{formatted_rewards}]")
        
        # 显示是否有效
        print(f"\n动作是否有效: {info['action_is_effective']}")
        
        # 显示目标点覆盖情况
        print("\n目标点覆盖情况:")
        for goal_idx, goal_pos in enumerate(env.goal_positions):
            if goal_idx in env.agents_on_goals:
                agent = env.agents_on_goals[goal_idx]
                print(f"目标 {goal_idx+1} 被 {agent} 覆盖")
            else:
                print(f"目标 {goal_idx+1} 未被覆盖")
        
        # 显示网格
        print("\n当前网格:")
        for line in env.render(mode='text'):
            print(line)
        
        # 检查是否结束
        if done:
            if env.success():
                print("\n所有目标点都有智能体!")
            elif env.step_count >= env.max_steps:
                print("\n已达到最大步数!")
    
    # 保存最终图像
    final_img = env.get_image('rgb_array')
    plt.imsave('multi_goal_nav_final.png', final_img)
    
    # 保存文本表示
    with open('multi_goal_nav_final_text.txt', 'w', encoding='utf-8') as f:
        text_grid = env.render(mode='text')
        f.write('\n'.join(text_grid))
        
        # 保存目标点完成情况
        f.write('\n\n=== 目标点覆盖情况 ===\n')
        for i, pos in enumerate(env.goal_positions):
            if i in env.agents_on_goals:
                agent = env.agents_on_goals[i]
                f.write(f"目标 {i+1}: {pos} - 被 {agent} 覆盖\n")
            else:
                f.write(f"目标 {i+1}: {pos} - 未被覆盖\n")
    
    print("\n最终状态已保存到multi_goal_nav_final.png和multi_goal_nav_final_text.txt")
    
    # 关闭环境
    env.close()