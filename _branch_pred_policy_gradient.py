import random

class PolicyGradientBranchPredictor:
    def __init__(self, ghr_bits, learning_rate):
        """
        Initialize the Policy Gradient Branch Predictor.
        :param ghr_bits: Number of Global History Register (GHR) bits.
        :param learning_rate: Learning rate for policy updates.
        """
        self.ghr_bits = ghr_bits  # Number of global history bits
        self.learning_rate = learning_rate  # Learning rate
        self.policy = {}  # Policy parameters θ[PC], initialized for each branch PC
        self.global_history = []  # Global History Register (GHR)
    
    def initialize_pc(self, pc):
        """
        Initialize policy parameters for a new program counter (PC).
        :param pc: Program counter (branch identifier).
        """
        if pc not in self.policy:
            self.policy[pc] = 0.0  # Start with θ[PC] = 0

    def get_features(self, pc, action):
        """
        Compute the feature vector x(s, a).
        :param pc: Program counter (branch identifier).
        :param action: The action taken ('T' or 'NT').
        :return: Feature vector based on GHR and PC.
        """
        # Combine GHR bits and PC to form a feature vector
        ghr_bits = ''.join(self.global_history[-self.ghr_bits:])  # Last `ghr_bits` from GHR
        feature_vector = f"{pc}_{ghr_bits}_{action}"
        return feature_vector

    def predict(self, pc):
        """
        Predict whether the branch will be taken ('T') or not taken ('NT').
        :param pc: Program counter (branch identifier).
        :return: Predicted action ('T' or 'NT').
        """
        self.initialize_pc(pc)
        # Predict action probabilistically based on current policy
        probability_taken = self.sigmoid(self.policy[pc])
        return 'T' if random.random() < probability_taken else 'NT'

    def update_policy(self, pc, action, reward):
        """
        Update the policy parameters based on the reward received.
        :param pc: Program counter (branch identifier).
        :param action: The action taken ('T' or 'NT').
        :param reward: The reward received (+1 for correct, -1 for incorrect).
        """
        self.initialize_pc(pc)
        # Compute feature vector and update θ[PC]
        features = self.get_features(pc, action)
        probability_action = self.sigmoid(self.policy[pc])
        gradient = reward * probability_action
        self.policy[pc] += self.learning_rate * gradient

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        :param x: Input value.
        :return: Sigmoid output (probability between 0 and 1).
        """
        return 1 / (1 + math.exp(-x))

    def update_global_history(self, outcome):
        """
        Update the Global History Register (GHR) with the branch outcome.
        :param outcome: Outcome of the branch ('T' or 'NT').
        """
        if len(self.global_history) >= self.ghr_bits:
            self.global_history.pop(0)  # Remove oldest history bit if GHR is full
        self.global_history.append('1' if outcome == 'T' else '0')


# Example Usage
if __name__ == "__main__":
    import math

    # Initialize the policy gradient branch predictor
    predictor = PolicyGradientBranchPredictor(ghr_bits=4, learning_rate=0.1)

    # Example program counter (PC) and branch outcomes
    branches = [
        {"pc": 1001, "outcome": 'T'},  # Branch 1
        {"pc": 1002, "outcome": 'NT'}, # Branch 2
        {"pc": 1001, "outcome": 'T'},  # Branch 1 again
        {"pc": 1003, "outcome": 'T'},  # Branch 3
    ]

    # Simulate branch prediction and policy updates
    for branch in branches:
        pc = branch["pc"]
        actual_outcome = branch["outcome"]

        # Predict the branch outcome
        predicted_outcome = predictor.predict(pc)

        # Reward: +1 for correct prediction, -1 for incorrect
        reward = 1 if predicted_outcome == actual_outcome else -1

        # Update policy based on reward
        predictor.update_policy(pc, predicted_outcome, reward)

        # Update Global History Register (GHR)
        predictor.update_global_history(actual_outcome)

        print(f"Branch PC: {pc}, Actual: {actual_outcome}, Predicted: {predicted_outcome}, Reward: {reward}")

    # Print the learned policy
    print("\nLearned Policy Parameters:")
    for pc, theta in predictor.policy.items():
        print(f"PC: {pc}, θ: {theta}")
