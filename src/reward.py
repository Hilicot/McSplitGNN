from options import opt

short_memory_threshold = 1e5
long_memory_threshold = 1e9


class Reward:

    def __init__(self):
        self.rl_component = 0
        self.ll_component = 0
        self.dal_component = 0
        self.normalized_reward = 0

    def get_reward(self, normalized: bool):
        if normalized:
            raise NotImplementedError("Normalized reward not implemented")
        if opt.reward_policy.current_reward_policy == opt.c.RL_POLICY:
            return self.rl_component
        elif opt.reward_policy.current_reward_policy == opt.c.DAL_POLICY:
            return self.ll_component + self.dal_component
        else:
            raise Exception("Invalid reward policy")

    def reset(self, value):
        self.rl_component = value
        self.ll_component = value
        self.dal_component = 0

    def decay(self):
        self.ll_component /= 2
        self.dal_component /= 2

    def update(self, ll_reward, dal_reward):
        self.rl_component += ll_reward
        self.ll_component += ll_reward
        self.dal_component += dal_reward


class DoubleQRewards:
    def __init__(self, n, m):
        self.V = [Reward() for _ in range(n)]
        self.Q = [[Reward() for _ in range(m)] for _ in range(n)]
        self.SingleQ = [Reward() for _ in range(m)]

    @staticmethod
    def rotate_reward_policy():
        opt.reward_policy.current_reward_policy = (opt.reward_policy.current_reward_policy + 1) \
                                                  %opt.reward_policy.reward_policies_num

    def reset_rewards(self):
        raise NotImplementedError("Not implemented")

    def randomize_rewards(self):
        raise NotImplementedError("Not implemented")

    def update_policy_counter(self, restart_counter: bool):
        if restart_counter:
            # A better solution was found, reset the counter
            opt.reward_policy.policy_switch_counter = 0
        else:
            # Increase the policy counter
            opt.reward_policy.policy_switch_counter += 1
            if opt.reward_policy.policy_switch_counter > opt.reward_policy.reward_switch_policy_threshold:
                opt.reward_policy.policy_switch_counter = 0
                switch_policy = opt.reward_policy.switch_policy
                if switch_policy == "NO_CHANGE":
                    # Do nothing
                    pass
                elif switch_policy == "CHANGE":
                    self.rotate_reward_policy()
                elif switch_policy == "RESET":
                    self.rotate_reward_policy()
                    self.reset_rewards()
                elif switch_policy == "RANDOM":
                    self.rotate_reward_policy()
                    self.randomize_rewards()
                elif switch_policy == "STEAL":
                    raise NotImplementedError("Not implemented")
                else:
                    raise Exception("Invalid reward switch policy")

    def update_rewards(self, new_domains_result, v: int, w: int) -> None:
        reward = new_domains_result.reward
        new_domains = new_domains_result.new_domains

        dal_reward = len(new_domains)

        # update rewards
        if reward > 0:

            self.V[v].update(reward, dal_reward)
            self.SingleQ[w].update(reward, dal_reward)
            self.Q[v][w].update(reward, dal_reward)

            # Do not decay if current policy is RL!
            if opt.mcs_method != RL_DAL or opt.reward_policy.current_reward_policy != opt.c.RL_POLICY:
                # TODO if we normalize, we might have to adjust the thresholds
                if self.get_vertex_reward(v, False) > short_memory_threshold:
                    for r in self.V:
                        r.decay()
                if self.get_pair_reward(v, w, False) > long_memory_threshold:
                    for r in self.Q[v]:
                        r.decay()

    def get_vertex_reward(self, v: int, normalized: bool) -> float:
        return self.V[v].get_reward(normalized)

    def get_pair_reward(self, v: int, w: int, normalized: bool) -> float:
        if opt.mcs_method == RL_DAL and opt.reward_policy.current_reward_policy == opt.c.RL_POLICY:
            return self.SingleQ[w].get_reward(normalized)
        else:
            return self.Q[v][w].get_reward(normalized)