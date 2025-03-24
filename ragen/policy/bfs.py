from .base import BasePolicy
from collections import deque
import pickle

class BFSPolicy(BasePolicy):
    def __init__(self, max_nodes=None):
        super().__init__()
        self.max_nodes = max_nodes  

    def select_action(self, observation, env=None):
        queue = deque([(pickle.dumps(env), [])])
        visited = {env.render(mode="state").tobytes()}
        nodes_visited = 0 

        while queue:
            if self.max_nodes is not None and nodes_visited >= self.max_nodes:
                return None
            nodes_visited += 1

            state, actions = queue.popleft()
            current_env = pickle.loads(state)

            if current_env.success():
                return actions if actions else 0

            for action in current_env.get_all_actions():
                env_copy = pickle.loads(state)
                env_copy.step(action)
                new_state = env_copy.render(mode="state").tobytes()

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((pickle.dumps(env_copy), actions + [action]))

        return None