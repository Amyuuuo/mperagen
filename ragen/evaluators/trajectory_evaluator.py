from ragen.utils.helpers import generate_trajectory, generate_trajectory_multienv
from ragen.utils import set_seed

class TrajectoryEvaluator:
    def __init__(self, env, policy, max_steps=100):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.results = []

    def evaluate(self, seed):
        self.env.reset(seed=seed)
        return generate_trajectory(self.env, self.policy, self.max_steps)

    def batch_evaluate(self, seeds, pool_size=8, mp=True):
        from multiprocessing import Pool
        from tqdm import tqdm
        if mp:
            with Pool(pool_size) as pool:
                results = list(tqdm(pool.imap(self.evaluate, seeds), total=len(seeds)))
        else:
            results = generate_trajectory_multienv(self.env, self.policy, seeds, self.max_steps)
        self.results = results
        
        return results

    def print_metrics(self):
        success_rate = sum(t[-1]['success'] for t in self.results) / len(self.results) * 100
        avg_steps = sum(len(t) for t in self.results) / len(self.results)
        print(f"Success rate: {success_rate:.1f}%, Average steps: {avg_steps:.2f}")
        return success_rate, avg_steps



