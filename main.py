#%% Load data
import pandas as pd

# 파일 경로 설정
file_path = "/Users/henry/Desktop/Virginia Tech/2024 FALL/Lectures/Emerging topics in CS/Project 2/rl-retail-sales/archive/sales.csv/sales.csv"

# CSV 파일 읽기
data_original = pd.read_csv(file_path)

#%% Check data
# 데이터 크기와 샘플 확인
print("Data Shape:", data_original.shape)
print(data_original.head())

# 결측치 확인
print("Missing Values:")
print(data_original.isnull().sum())

# 데이터 타입 확인
print("\nData Types:")
print(data_original.dtypes)

#%% Drop columns
columns_to_drop = ['store_id', 'promo_type_1', 'promo_bin_1', 'promo_type_2', 'promo_bin_2', 'promo_discount_2', 'promo_discount_type_2']
data_dropped = data_original.drop(columns=columns_to_drop)

#%% Missing Values
# 결측치 있는 행 제외
print(data_dropped.isnull().sum())
data_cleaned = data_dropped.dropna()
print(data_cleaned.isnull().sum())

#%%
data_cleaned['date'] = pd.to_datetime(data_cleaned['date'])
print(data_cleaned.describe().T)

#%% 일 단위로 프로덕트 판매 수량 통일
data_daily = (
    data_cleaned.groupby(["date", "product_id"])
    .agg({
        'sales': 'sum',
        'revenue': 'sum',
        'stock': 'sum',     
        'price': 'first' 
    })
    .reset_index()  # 인덱스를 초기화하여 보기 쉽게
)

#%%
# 데이터 분할

# Data preprocessing
data_product_reduced =data_daily[data_daily['product_id'].isin(['P0001'])]

train_end_date = '2019-06-30'
val_end_date = '2019-08-31'

train_data = data_product_reduced[data_product_reduced['date'] <= train_end_date]
val_data = data_product_reduced[(data_product_reduced['date'] > train_end_date) & (data_daily['date'] <= val_end_date)]
test_data = data_product_reduced[data_product_reduced['date'] > val_end_date]

product_ids_train = train_data['product_id'].unique()
product_ids_val = val_data['product_id'].unique()
product_ids_test = test_data['product_id'].unique()

print(f"Train data: {train_data.shape}")
print(f"Validation data: {val_data.shape}")
print(f"Test data: {test_data.shape}")

# 확인
print(f"Training data: {train_data['date'].min()} ~ {train_data['date'].max()}") 
print(f"Validation data: {val_data['date'].min()} ~ {val_data['date'].max()}")
print(f"Test data: {test_data['date'].min()} ~ {test_data['date'].max()}")


#%%


import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from scipy.stats import entropy
from scipy.special import softmax

# === Environment Setup ===
class RetailEnv(gym.Env):
    def __init__(self, data, product_ids, max_stock=1000, max_order=30, stock_penalty=300, no_stock_penalty=300):
        super(RetailEnv, self).__init__()
        self.data = data
        self.product_ids = product_ids
        self.max_stock = max_stock
        self.max_order = max_order
        self.stock_penalty = stock_penalty
        self.no_stock_penalty = no_stock_penalty
        self.uncertainties = {'vacuity': [], 'dissonance': [], 'entropy': []}

        # Define action and state spaces
        self.action_space = gym.spaces.Box(low=0, high=max_order, shape=(len(product_ids),), dtype=np.int32)
        self.observation_space = gym.spaces.Box(
            low=0, high=max_stock, shape=(len(product_ids), 3), dtype=np.float32  # stock, sales, price
        )

        # Initialize state
        self.current_step = 0
        self.state = None

    def reset(self):
        self.current_step = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        current_date = self.data.iloc[self.current_step]['date']  # 현재 날짜 가져오기
        current_data = self.data[self.data['date'] == current_date]  # 해당 날짜의 데이터 필터링

        state = []
        for product_id in self.product_ids:
            if product_id in current_data['product_id'].values:
                product_row = current_data[current_data['product_id'] == product_id]
                stock = product_row['stock'].values[0]
                sales = product_row['sales'].values[0]
                price = product_row['price'].values[0]
            else:
                stock, sales, price = 0, 0, 0  # Missing product
            state.append([stock, sales, price])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        rewards = 0
        done = False

        # Calculate reward and update state
        for idx, product_id in enumerate(self.product_ids):
            if idx >= len(action):
                continue

            current_stock = self.state[idx][0]
            sales = self.state[idx][1]
            price = self.state[idx][2]
            order_quantity = action[idx]

            new_stock = current_stock + order_quantity - sales
            new_stock = max(0, min(new_stock, self.max_stock))  # Clip to valid range

            # 불확실성 계산
            q_values = np.array([0.3, 0.5, 0.2])  # 임의의 예측값
            q_values = softmax(q_values)
            alpha = q_values * 10
            dirichlet_sample = dirichlet.rvs(alpha, size=1).flatten()
            opinion = compute_multinomial_opinion(dirichlet_sample) # 확신을 갖는 정도

            rewards -= np.sum(opinion) * 5  # Opinion에서 값의 합을 페널티로 사용            
            
            # 1. 판매량에 따른 보상
            rewards += sales * price  # 매출 보상

            # 2. 최적 주문량 보상
            if abs(order_quantity - sales) <= self.max_order * 0.1:  # 주문량과 판매량의 차이가 적으면 추가 보상
                rewards += 20

            # 3. 페널티: 재고 부족
            if new_stock <= 30:
                rewards -= self.no_stock_penalty

            # 4. 페널티: 과도한 재고
            elif new_stock > self.max_stock * 0.8:
                rewards -= self.stock_penalty

            # 5. 재고 유지 비용
            rewards -= new_stock * 0.01  # 재고 1개당 유지 비용
            
            # 불확실성 기반 보상
            vacuity = 1 - np.sum(dirichlet_sample) / len(dirichlet_sample) # Dirichlet 샘플이 전체적으로 작아서, 모델이 확신이 없을 때
            dissonance = np.max(dirichlet_sample) - np.min(dirichlet_sample) # Dissonance는 Dirichlet 샘플 내에서 가장 큰 값과 작은 값의 차이
            entropy_value = entropy(dirichlet_sample) # Dirichlet 샘플의 전체적인 분산도
            
            self.uncertainties['vacuity'].append(vacuity)
            self.uncertainties['dissonance'].append(dissonance)
            self.uncertainties['entropy'].append(entropy_value)
            
            if vacuity > 0.5: # 불확실성이 매우 높다는 뜻
                rewards -= 10
            if dissonance > 0.4: # 불균형한 신뢰도
                rewards -= 10
            if entropy_value > 0.7: # 행동 간의 차별성이 적다는 뜻으로, 불확실성이 높음
                rewards -= 10

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
        else:
            self.state = self._get_state()

        return self.state, rewards, done, {}


# === DQN Agent ===
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, epsilon_decay=0.995, gamma=0.99, max_order=30):
        self.state_size = state_size
        self.action_size = action_size  # This should be len(product_ids), not multiplied by max_order
        self.max_order = max_order
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Neural network for Q-learning
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),  # The output size should be len(product_ids)
            nn.Sigmoid()
        )
        return model
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.max_order, self.action_size)  # Random action for each product
        else:            
            state = torch.tensor(state.flatten(), dtype=torch.float32)
            q_values = self.model(state).detach().numpy()  # Output Q-values for each product
            q_values = softmax(q_values)
            alpha = q_values * 10
            dirichlet_sample = dirichlet.rvs(alpha, size=1).flatten()
            opinion = compute_multinomial_opinion(dirichlet_sample)

            action = (dirichlet_sample * self.max_order).astype(int)
            action = np.clip(action, 0, self.max_order)
            
            return action


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state.flatten(), dtype=torch.float32)
                target += self.gamma * torch.max(self.model(next_state)).item()
            state = torch.tensor(state.flatten(), dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.long)
            prediction = self.model(state)
            
            # Ensure target is a tensor with the same shape as prediction
            target = torch.tensor(target, dtype=torch.float32).view(-1)  # Ensure target is a tensor with the correct shape
                      
            # 손실 계산 및 역전파
            loss = nn.MSELoss()(prediction, target)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return np.mean(losses)

# === Training and Validation Functions ===
def train_and_validate(train_env, agent, val_env, episodes, batch_size=32):
    episode_rewards = []
    episode_losses = []
    episode_uncertainties = {'vacuity': [], 'dissonance': [], 'entropy': []}
    
    for e in range(episodes):
        state = train_env.reset()
        total_reward = 0
        total_loss = 0
        train_env.uncertainties = {'vacuity': [], 'dissonance': [], 'entropy': []}

        for time in range(len(train_env.data)):
            action = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward            
            
            if time % 100 == 0:
                loss = agent.replay(batch_size)
            if loss is not None:
                total_loss += loss
            
            if time % 100 == 0:  # 10번마다 로그 출력
                print(f"Episode {e + 1}/{episodes}, Step {time}, Reward: {reward}, Total Reward: {total_reward}")
            
            if done:
                break
        
        for metric in episode_uncertainties.keys():
                episode_uncertainties[metric].append(np.mean(train_env.uncertainties[metric]))
            
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss / (time + 1))
        agent.replay(batch_size)
        
        # Log uncertainties per episode
        avg_vacuity = np.mean(train_env.uncertainties['vacuity'])
        avg_dissonance = np.mean(train_env.uncertainties['dissonance'])
        avg_entropy = np.mean(train_env.uncertainties['entropy'])

        print(f"Episode {e + 1} - Vacuity: {avg_vacuity:.4f}, Dissonance: {avg_dissonance:.4f}, Entropy: {avg_entropy:.4f}")
        
        print(f"Episode {e + 1} finished with Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
    
    return episode_rewards, episode_losses, episode_uncertainties


# Validation function
def validate_model(val_env, agent):
    state = val_env.reset()
    total_val_reward = 0
    for _ in range(len(val_env.data)):
        action = agent.act(state)
        next_state, reward, done, _ = val_env.step(action)
        state = next_state
        total_val_reward += reward
        if done:
            break
    return total_val_reward

def compute_multinomial_opinion(dirichlet_sample):
    opinion = dirichlet_sample / np.sum(dirichlet_sample)  # Normalize to form a probability distribution
    return opinion

# === Model Saving Function ===
def save_model(agent, filepath="/Users/henry/Desktop/Virginia Tech/2024 FALL/Lectures/Emerging topics in CS/Project 2/rl-retail-sales/dqn_retail_model.pth"):
    torch.save(agent.model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

# 모델 성능 그래프 그리기
def plot_performance(validation_rewards):
    plt.figure(figsize=(10, 6))
    models = [f'Model {i+1}' for i in range(len(validation_rewards))]
    
    plt.bar(models, validation_rewards, color='skyblue')
    plt.title('Validation Rewards for Different Models')
    plt.xlabel('Models')
    plt.ylabel('Total Validation Reward')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_uncertainties(uncertainties_log):
    plt.figure(figsize=(12, 8))
    for metric, values in uncertainties_log.items():
        plt.plot(values, label=f'{metric.capitalize()}')
    plt.title('Uncertainty Metrics Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid()
    plt.show()


# === Main Function ===
# Create environments
train_env = RetailEnv(train_data, product_ids_train)
val_env = RetailEnv(val_data, product_ids_val)
test_env = RetailEnv(test_data, product_ids_test)

# Create multiple models with different hyperparameters
hyperparameters = [
    {'learning_rate': 0.001, 'epsilon_decay': 0.995, 'gamma': 0.99},
    {'learning_rate': 0.005, 'epsilon_decay': 0.998, 'gamma': 0.95},
    {'learning_rate': 0.0005, 'epsilon_decay': 0.990, 'gamma': 0.90},
    {'learning_rate': 0.002, 'epsilon_decay': 0.997, 'gamma': 0.98}
]

models = []
validation_rewards = []
model_episode_rewards = []
model_episode_losses = []

# Train models with different hyperparameters
for i, params in enumerate(hyperparameters):
    agent = DQNAgent(
        state_size=train_env.observation_space.shape[0] * train_env.observation_space.shape[1], 
        action_size=len(train_env.product_ids),  # action_size is len(product_ids)
        **params
    )
    print(f"Training model {i + 1} with parameters: {params}")
    
    rewards, losses, uncertainties_log = train_and_validate(train_env, agent, val_env, episodes=500, batch_size=32)
    model_episode_rewards.append(rewards)
    model_episode_losses.append(losses)
    
    models.append(agent)
    
    # Track model performance
    val_reward = validate_model(val_env, agent)
    validation_rewards.append(val_reward)


# 모델 성능 그래프 그리기
plot_performance(validation_rewards)
# 불확실성 그래프 그리기
plot_uncertainties(uncertainties_log)

# Save all models with hyperparameters in filename
for i, (agent, params) in enumerate(zip(models, hyperparameters)):
    param_str = f"lr_{params['learning_rate']}_ed_{params['epsilon_decay']}_g_{params['gamma']}"
    save_path = f"/Users/henry/Desktop/Virginia Tech/2024 FALL/Lectures/Emerging topics in CS/Project 2/rl-retail-sales/dqn_retail_model_{i + 1}_{param_str}.pth"
    save_model(agent, filepath=save_path)
    print(f"Model {i + 1} saved with hyperparameters: {param_str}, Validation Reward = {validation_rewards[i]}")

# Select the best model based on validation reward
best_model_idx = np.argmax(validation_rewards)
best_agent = models[best_model_idx]
print(f"Best Model: Model {best_model_idx + 1} with Validation Reward = {validation_rewards[best_model_idx]}")

test_reward = validate_model(test_env, best_agent)
print(f"Test Reward from Model {best_model_idx + 1} = {test_reward}")

#%%
# 에피소드 보상 그래프 그리기
plt.figure(figsize=(12, 8))
for i, rewards in enumerate(model_episode_rewards):
    plt.plot(rewards, label=f'Model {i+1}')
plt.title('Training Rewards Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.legend()
plt.show()

#%%
# 에피소드 손실 그래프 그리기
plt.figure(figsize=(12, 8))
for i, losses in enumerate(model_episode_losses):
    plt.plot(losses, label=f'Model {i+1}')
plt.title('Average Loss Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Average Loss')
plt.legend()
plt.show()

#%%
# Effectiveness 측정: Validation 데이터에서 Total Reward 계산
def measure_effectiveness(env, agent):
    state = env.reset()
    total_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward

# Efficiency 측정: 훈련 시의 평균 손실 계산
def measure_efficiency(losses):
    return np.mean(losses)

# Best Agent 선정 및 측정
best_agent_index = np.argmax(validation_rewards)  # Validation Reward가 가장 높은 모델
best_agent = models[best_agent_index]

# Validation 환경에서 Effectiveness 계산
effectiveness = measure_effectiveness(val_env, best_agent)
print(f"Effectiveness (Total Reward on Validation): {effectiveness}")

# Training 손실로 Efficiency 계산
efficiency = measure_efficiency(model_episode_losses[best_agent_index])
print(f"Efficiency (Mean Loss during Training): {efficiency}")

#%%
def sensitivity_analysis(max_order_values, train_env, val_env, episodes=100, batch_size=32):
    sensitivity_results = {}
    
    for max_order in max_order_values:
        print(f"\nRunning sensitivity analysis for max_order={max_order}")
        # Update the environment's max_order
        train_env.max_order = max_order
        val_env.max_order = max_order
        
        # Create a new agent for each max_order value
        agent = DQNAgent(
            state_size=train_env.observation_space.shape[0] * train_env.observation_space.shape[1],
            action_size=len(train_env.product_ids),
            max_order=max_order
        )
        
        # Train the agent
        rewards, losses, uncertainties_log = train_and_validate(train_env, agent, val_env, episodes, batch_size)
        
        # Validate the agent
        val_reward = validate_model(val_env, agent)
        sensitivity_results[max_order] = {
            'val_reward': val_reward,
            'rewards': rewards,
            'losses': losses,
            'uncertainties': uncertainties_log
        }
    
    return sensitivity_results

def plot_sensitivity_analysis(sensitivity_results):
    # Extract results
    max_orders = list(sensitivity_results.keys())
    val_rewards = [sensitivity_results[mo]['val_reward'] for mo in max_orders]
    
    # Plot Validation Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(max_orders, val_rewards, marker='o', label='Validation Reward')
    plt.title('Sensitivity Analysis: Effect of max_order on Performance')
    plt.xlabel('Max Order')
    plt.ylabel('Validation Reward')
    plt.xticks(max_orders)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot Training Rewards Over Episodes
    plt.figure(figsize=(12, 8))
    for max_order, results in sensitivity_results.items():
        plt.plot(results['rewards'], label=f'max_order={max_order}')
    plt.title('Training Rewards Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Run sensitivity analysis
max_order_values = [30, 50, 100]
sensitivity_results = sensitivity_analysis(max_order_values, train_env, val_env, episodes=100, batch_size=32)

# Plot results
plot_sensitivity_analysis(sensitivity_results)







#%%
# === 민감도 분석 및 결과 시각화 ===

def sensitivity_analysis_max_stock(max_stock_values, train_data, val_data, product_ids_train, product_ids_val, hyperparameters):
    results = {}  # 각 max_stock에 대한 결과 저장
    for max_stock in max_stock_values:
        print(f"\nRunning sensitivity analysis for max_stock = {max_stock}")
        
        # 새로운 환경 생성
        train_env = RetailEnv(train_data, product_ids_train, max_stock=max_stock)
        val_env = RetailEnv(val_data, product_ids_val, max_stock=max_stock)
        
        # 모델 초기화 및 학습
        model_rewards = []
        for i, params in enumerate(hyperparameters):
            agent = DQNAgent(
                state_size=train_env.observation_space.shape[0] * train_env.observation_space.shape[1], 
                action_size=len(train_env.product_ids),  # action_size is len(product_ids)
                **params
            )
            print(f"  Training model {i + 1} with parameters: {params}")
            
            # 학습 및 검증
            rewards, _, _ = train_and_validate(train_env, agent, val_env, episodes=100, batch_size=32)
            val_reward = validate_model(val_env, agent)
            model_rewards.append(val_reward)
        
        # 민감도 분석 결과 저장
        results[max_stock] = model_rewards
    
    return results


def plot_sensitivity_analysis_max_stock(results, max_stock_values):
    plt.figure(figsize=(10, 6))
    for max_stock, rewards in results.items():
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Max Stock = {max_stock}")
    
    plt.title("Sensitivity Analysis: Validation Rewards for Different Max Stock Levels")
    plt.xlabel("Model Index")
    plt.ylabel("Validation Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# 민감도 분석 실행
max_stock_values = [500, 1000, 2000]
sensitivity_results = sensitivity_analysis_max_stock(
    max_stock_values=max_stock_values,
    train_data=train_data,
    val_data=val_data,
    product_ids_train=product_ids_train,
    product_ids_val=product_ids_val,
    hyperparameters=hyperparameters
)

# 민감도 분석 결과 시각화
plot_sensitivity_analysis_max_stock(sensitivity_results, max_stock_values)




#%%


#%%
# === Load Model Function ===
def load_model(agent, filepath="/Users/henry/Desktop/Virginia Tech/2024 FALL/Lectures/Emerging topics in CS/Project 2/rl-retail-sales/first results/dqn_retail_model_2_lr_0.005_ed_0.998_g_0.95.pth"):
    agent.model.load_state_dict(torch.load(filepath))
    agent.model.eval()  # 평가 모드로 전환
    print(f"Model loaded from {filepath}")

base_agent =load_model(agent, filepath="/Users/henry/Desktop/Virginia Tech/2024 FALL/Lectures/Emerging topics in CS/Project 2/rl-retail-sales/first results/dqn_retail_model_2_lr_0.005_ed_0.998_g_0.95.pth")

# Validation 환경에서 Effectiveness 계산
effectiveness = measure_effectiveness(val_env, base_agent)
print(f"Effectiveness (Total Reward on Validation): {effectiveness}")
