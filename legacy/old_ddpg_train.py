import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from google.colab import drive

# 挂载 Google Drive
if not os.path.exists('/content/drive'):
drive.mount('/content/drive')

print("=======================================================")
print("🛡️ Colab 加速防弹版：连续分数 + 智能寻址 + 快速训练")
print("=======================================================")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 使用设备: {device}")

# ==========================================
# 1. 🔍 智能雷达寻址（自动在 Drive 里找所有必需文件）
# ==========================================
base_path = None
for root_dir, dirs, files in os.walk('/content/drive'):
if 'q_matrix_multihot_36_expert.pt' in files:
base_path = root_dir
break

if base_path is None:
raise FileNotFoundError("❌ 在 Google Drive 中没找到 q_matrix_multihot_36_expert.pt！")

print(f"✅ 雷达寻址成功！资产目录: {base_path}")

multi_hot_q_path = f'{base_path}/q_matrix_multihot_36_expert.pt'
ncdm_36d_save_path = f'{base_path}/ncdm_model_36d_expert_best.pt'
solidified_128_path = f'{base_path}/item_bank_128d.npy'

# 搜索 CSV 文件（可能在 kc_level 或其他目录）
train_csv_path = None
test_csv_path = None
for root_dir, dirs, files in os.walk('/content/drive'):
if 'train_valid_sequences.csv' in files and train_csv_path is None:
train_csv_path = os.path.join(root_dir, 'train_valid_sequences.csv')
if 'test.csv' in files and test_csv_path is None:
test_csv_path = os.path.join(root_dir, 'test.csv')
if train_csv_path and test_csv_path:
break

if train_csv_path is None:
raise FileNotFoundError("❌ 未找到 train_valid_sequences.csv")
if test_csv_path is None:
raise FileNotFoundError("❌ 未找到 test.csv")

print(f"✅ 训练数据: {train_csv_path}")
print(f"✅ 测试数据: {test_csv_path}")

# 加载静态高维资产
q_masks_tensor = torch.load(multi_hot_q_path).to(device)
K_DIM = 36
num_items = q_masks_tensor.shape[0]

frozen_128d_bank = torch.tensor(np.load(solidified_128_path), dtype=torch.float32).to(device)
frozen_128d_bank = nn.functional.normalize(frozen_128d_bank, p=2, dim=1)
MAX_SAFE_ITEM_ID = min(num_items, frozen_128d_bank.shape[0])

# ==========================================
# 🛠️ 智能数据嗅探与列名锁定
# ==========================================
df_ncdm_train = pd.read_csv(train_csv_path)
unique_uids = df_ncdm_train['uid'].unique()
uid_to_idx = {uid: idx for idx, uid in enumerate(unique_uids)}
num_users = len(uid_to_idx)

print(f"\n🔍 自动探测到 CSV 拥有以下列名: {df_ncdm_train.columns.tolist()}")
q_col = next((c for c in df_ncdm_train.columns if 'question' in c.lower() or 'item' in c.lower()), df_ncdm_train.columns[1])
r_col = next((c for c in df_ncdm_train.columns if 'response' in c.lower() or 'score' in c.lower() or 'correct' in c.lower()), df_ncdm_train.columns[2])
print(f"✅ 已成功智能锁定 -> 题目特征列: [{q_col}], 学生得分列: [{r_col}]")

def get_clean_sequence(row_q, row_r):
q_strs = str(row_q).replace('[', '').replace(']', '').replace("'", "").replace('"', '').split(',')
r_strs = str(row_r).replace('[', '').replace(']', '').replace("'", "").replace('"', '').split(',')
v_i, v_r = [], []
for q, r in zip(q_strs, r_strs):
if not q.strip() or not r.strip(): continue
q_id = int(float(q.strip()))
r_val = float(r.strip())
if 0 <= q_id < MAX_SAFE_ITEM_ID and r_val >= 0.0:
v_i.append(q_id)
v_r.append(r_val)
return v_i, v_r

# ==========================================
# 2. 🧠 官方 NCDM 判官（智能规避维度冲突）
# ==========================================
class NoneNegClipper(object):
def __init__(self):
super(NoneNegClipper, self).__init__()
def __call__(self, module):
if hasattr(module, 'weight'):
w = module.weight.data
w.add_(torch.relu(torch.neg(w)))

class OfficialNCDM(nn.Module):
def __init__(self, student_n, exer_n, knowledge_n):
super(OfficialNCDM, self).__init__()
self.knowledge_dim = knowledge_n
self.exer_n = exer_n
self.emb_num = student_n
self.prednet_len1 = 512
self.prednet_len2 = 256 
self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
self.e_discrimination = nn.Embedding(self.exer_n, 1)
self.prednet_full1 = nn.Linear(self.knowledge_dim, self.prednet_len1)
self.drop_1 = nn.Dropout(p=0.5)
self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
self.drop_2 = nn.Dropout(p=0.5)
self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

def forward(self, stu_id, exer_id, kn_emb):
stu_emb = torch.sigmoid(self.student_emb(stu_id))
k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
output = torch.sigmoid(self.prednet_full3(input_x))
return output.squeeze(-1)

def apply_clipper(self):
clipper = NoneNegClipper()
self.prednet_full1.apply(clipper)
self.prednet_full2.apply(clipper)
self.prednet_full3.apply(clipper)

ncdm_doctor = OfficialNCDM(num_users, num_items, K_DIM).to(device)

# 智能安全加载判官参数
checkpoint = torch.load(ncdm_36d_save_path, map_location=device)
if 'student_emb.weight' in checkpoint:
del checkpoint['student_emb.weight']
ncdm_doctor.load_state_dict(checkpoint, strict=False)
print("✅ 已成功避开旧模型学生维度冲突，题目判官参数加载完毕！")

ncdm_doctor.apply_clipper()
ncdm_doctor.eval()
for param in ncdm_doctor.parameters():
param.requires_grad = False

# 🚀 极速 Alpha 追踪引擎（8步拟合）
def evaluate_entropy_ncdm(history_i, history_r, target_i):
if len(history_i) == 0:
alpha = torch.zeros((1, K_DIM), device=device, requires_grad=True)
else:
h_t_i = torch.tensor(history_i, dtype=torch.long, device=device)
h_t_r = torch.tensor(history_r, dtype=torch.float, device=device)
with torch.no_grad():
hist_diffs = torch.sigmoid(ncdm_doctor.k_difficulty(h_t_i))
mean_diff = hist_diffs.mean(dim=0).clamp(1e-4, 1 - 1e-4)
init_val = torch.log(mean_diff / (1.0 - mean_diff)) 
alpha = init_val.unsqueeze(0).clone().detach().requires_grad_(True)
opt = optim.Adam([alpha], lr=0.05)
if len(history_i) > 0:
for step_idx in range(8):
opt.zero_grad()
diff = torch.sigmoid(ncdm_doctor.k_difficulty(h_t_i))
disc = torch.sigmoid(ncdm_doctor.e_discrimination(h_t_i)) * 10
input_x = disc * (torch.sigmoid(alpha) - diff) * q_masks_tensor[h_t_i]
input_x = ncdm_doctor.drop_1(torch.sigmoid(ncdm_doctor.prednet_full1(input_x)))
input_x = ncdm_doctor.drop_2(torch.sigmoid(ncdm_doctor.prednet_full2(input_x)))
pred = torch.sigmoid(ncdm_doctor.prednet_full3(input_x)).squeeze(-1)
loss = nn.BCELoss()(pred, h_t_r)
loss.backward()
opt.step()
if step_idx >= 5 and loss.item() < 1e-3:
break
with torch.no_grad():
if not target_i:
return 0.0, alpha.detach()
t_t_i = torch.tensor(target_i, dtype=torch.long, device=device)
diff = torch.sigmoid(ncdm_doctor.k_difficulty(t_t_i))
disc = torch.sigmoid(ncdm_doctor.e_discrimination(t_t_i)) * 10
input_x = disc * (torch.sigmoid(alpha) - diff) * q_masks_tensor[t_t_i]
input_x = ncdm_doctor.drop_1(torch.sigmoid(ncdm_doctor.prednet_full1(input_x)))
input_x = ncdm_doctor.drop_2(torch.sigmoid(ncdm_doctor.prednet_full2(input_x)))
p = torch.sigmoid(ncdm_doctor.prednet_full3(input_x)).squeeze(-1)
p = torch.clamp(p, 1e-5, 1.0 - 1e-5)
ent = - (p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
mean_entropy = ent.mean().item()
return mean_entropy, alpha.detach()

# ==========================================
# 3. 🚀 连续分数 Actor-Critic 智能体
# ==========================================
class LSTMActor(nn.Module):
def __init__(self, semantic_dim=128, q_dim=36, resp_dim=32, hidden_dim=128):
super().__init__()
self.response_emb = nn.Linear(1, resp_dim)
input_dim = semantic_dim + q_dim + 36 + 1 + resp_dim
self.norm = nn.LayerNorm(input_dim)
self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
self.policy_head = nn.Sequential(
nn.Linear(hidden_dim, 256), nn.ReLU(),
nn.Linear(256, 73), nn.Sigmoid() 
)
def forward(self, semantic_vec, q_mask_vec, diff_vec, disc_vec, response_val, hx, cx):
resp_f = torch.relu(self.response_emb(response_val.unsqueeze(-1)))
x_t = torch.cat([semantic_vec, q_mask_vec, diff_vec, disc_vec, resp_f], dim=-1)
x_t = self.norm(x_t)
hx, cx = self.lstm_cell(x_t, (hx, cx))
ideal_73d_vector = self.policy_head(hx)
return ideal_73d_vector, hx, cx

class LSTMCritic(nn.Module):
def __init__(self, hidden_dim=128, action_dim=73): 
super().__init__()
self.critic_net = nn.Sequential(
nn.Linear(hidden_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1)
)
def forward(self, hx, action_vector):
x = torch.cat([hx, action_vector], dim=-1)
return torch.clamp(self.critic_net(x), -2000.0, 2000.0)

actor = LSTMActor().to(device)
critic = LSTMCritic().to(device)
target_actor = LSTMActor().to(device)
target_critic = LSTMCritic().to(device)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

# ==========================================
# 4. 🔥 快速训练配置（5 Epochs, 300 学生, 10 步）
# ==========================================
memory = deque(maxlen=20000)
batch_size = 64
gamma = 0.99
tau = 0.005 
train_students = df_ncdm_train.sample(n=min(300, len(df_ncdm_train)), random_state=42)

print("\n🔥 显卡已全负荷挂载，开始强化学习矩阵演进...")
for epoch in range(5):
epoch_reward, valid_count, epoch_c_loss, batch_count = 0, 0, 0, 0
noise_std = max(0.2, 0.5 - (epoch / 10.0) * 0.45)
for _, row in tqdm(train_students.iterrows(), total=len(train_students), desc=f"Epoch {epoch+1}/5"):
v_i, v_r = get_clean_sequence(row[q_col], row[r_col])
if len(v_i) < 16: continue
valid_count += 1
indices = list(range(len(v_i)))
random.shuffle(indices)
split_idx = int(len(v_i) * 0.7)
avail_i = [v_i[i] for i in indices[:split_idx]]
avail_r = [v_r[i] for i in indices[:split_idx]]
val_i = [v_i[i] for i in indices[split_idx:]]
hx = torch.zeros(1, 128, device=device)
cx = torch.zeros(1, 128, device=device)
seed_idx = random.choice(range(len(avail_i)))
current_item_id = avail_i.pop(seed_idx)
current_response_val = avail_r.pop(seed_idx)
history_i = [current_item_id]
history_r = [current_response_val]

for step in range(10):
if not avail_i: break
prev_entropy, _ = evaluate_entropy_ncdm(history_i, history_r, val_i)
curr_sem = frozen_128d_bank[current_item_id]
curr_q = q_masks_tensor[current_item_id]
with torch.no_grad():
c_t = torch.tensor([current_item_id], device=device)
curr_diff = torch.sigmoid(ncdm_doctor.k_difficulty(c_t)).squeeze(0)
curr_disc = torch.sigmoid(ncdm_doctor.e_discrimination(c_t)).squeeze(0)
actor.eval()
with torch.no_grad():
ideal_73d_vector, next_hx, next_cx = actor(
curr_sem.unsqueeze(0), curr_q.unsqueeze(0), curr_diff.unsqueeze(0), curr_disc.unsqueeze(0),
torch.tensor([current_response_val], dtype=torch.float, device=device), hx, cx
)
ideal_73d_vector = torch.clamp(ideal_73d_vector + torch.randn_like(ideal_73d_vector) * noise_std, 0.0, 1.0)
actor.train()

with torch.no_grad():
a_t = torch.tensor(avail_i, device=device)
candidate_73d = torch.cat([q_masks_tensor[avail_i], torch.sigmoid(ncdm_doctor.k_difficulty(a_t)), torch.sigmoid(ncdm_doctor.e_discrimination(a_t))], dim=-1)
chosen_local_idx = torch.argmin(torch.cdist(ideal_73d_vector, candidate_73d).squeeze(0)).item()
next_item_id = avail_i.pop(chosen_local_idx)
next_response_val = avail_r.pop(chosen_local_idx)
history_i.append(next_item_id)
history_r.append(next_response_val)
curr_entropy, _ = evaluate_entropy_ncdm(history_i, history_r, val_i)
reward = np.clip((prev_entropy - curr_entropy) * 50.0, -10.0, 10.0)
epoch_reward += reward
done = (step == 9) or (len(avail_i) == 0)
with torch.no_grad():
n_t = torch.tensor([next_item_id], device=device)
next_diff = torch.sigmoid(ncdm_doctor.k_difficulty(n_t)).squeeze(0)
next_disc = torch.sigmoid(ncdm_doctor.e_discrimination(n_t)).squeeze(0)
memory.append((
hx.squeeze(0).cpu().numpy(), cx.squeeze(0).cpu().numpy(),
curr_sem.cpu().numpy(), curr_q.cpu().numpy(), curr_diff.cpu().numpy(), curr_disc.cpu().numpy(), current_response_val,
ideal_73d_vector.squeeze(0).cpu().numpy(), reward,
frozen_128d_bank[next_item_id].cpu().numpy(), q_masks_tensor[next_item_id].cpu().numpy(), next_diff.cpu().numpy(), next_disc.cpu().numpy(), next_response_val,
done
))
hx, cx = next_hx.detach(), next_cx.detach() 
current_item_id = next_item_id
current_response_val = next_response_val
if done: break
if len(memory) > batch_size:
batch = random.sample(memory, batch_size)
b_h0 = torch.tensor(np.array([m[0] for m in batch]), dtype=torch.float, device=device)
b_c0 = torch.tensor(np.array([m[1] for m in batch]), dtype=torch.float, device=device)
b_sem = torch.tensor(np.array([m[2] for m in batch]), dtype=torch.float, device=device)
b_q = torch.tensor(np.array([m[3] for m in batch]), dtype=torch.float, device=device)
b_diff = torch.tensor(np.array([m[4] for m in batch]), dtype=torch.float, device=device)
b_disc = torch.tensor(np.array([m[5] for m in batch]), dtype=torch.float, device=device)
b_resp = torch.tensor(np.array([m[6] for m in batch]), dtype=torch.float, device=device)
b_act = torch.tensor(np.array([m[7] for m in batch]), dtype=torch.float, device=device)
b_rew = torch.tensor(np.array([m[8] for m in batch]), dtype=torch.float, device=device).unsqueeze(1)
b_next_sem = torch.tensor(np.array([m[9] for m in batch]), dtype=torch.float, device=device)
b_next_q = torch.tensor(np.array([m[10] for m in batch]), dtype=torch.float, device=device)
b_next_diff = torch.tensor(np.array([m[11] for m in batch]), dtype=torch.float, device=device)
b_next_disc = torch.tensor(np.array([m[12] for m in batch]), dtype=torch.float, device=device)
b_next_resp = torch.tensor(np.array([m[13] for m in batch]), dtype=torch.float, device=device)
b_done = torch.tensor(np.array([m[14] for m in batch]), dtype=torch.float, device=device).unsqueeze(1)
ideal_acts, b_h1, b_c1 = actor(b_sem, b_q, b_diff, b_disc, b_resp, b_h0, b_c0)
with torch.no_grad():
next_ideal_acts, b_nhx_target, _ = target_actor(b_next_sem, b_next_q, b_next_diff, b_next_disc, b_next_resp, b_h1.detach(), b_c1.detach())
y_expected = b_rew + gamma * target_critic(b_nhx_target, next_ideal_acts) * (1 - b_done)
critic_loss = nn.SmoothL1Loss()(critic(b_h1.detach(), b_act), y_expected)
critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
critic_optimizer.step()
epoch_c_loss += critic_loss.item()
batch_count += 1
actor_loss = -critic(b_h1, ideal_acts).mean()
actor_optimizer.zero_grad()
actor_loss.backward()
torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
actor_optimizer.step()
for p, tp in zip(actor.parameters(), target_actor.parameters()):
tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
for p, tp in zip(critic.parameters(), target_critic.parameters()):
tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

print(f"✅ Epoch {epoch+1} | 均值信息增益: {epoch_reward/max(1, valid_count*10):.4f} | Critic Loss: {epoch_c_loss/max(1, batch_count):.4f}")
torch.save(actor.state_dict(), f'{base_path}/ddpg_whitebox_36d_actor_colab.pt')

# ==========================================
# 5. 🎯 终极全样本盲测推演
# ==========================================
print("\n🏆 进入终极盲测推演对决...")
actor.load_state_dict(torch.load(f'{base_path}/ddpg_whitebox_36d_actor_colab.pt', map_location=device))
actor.eval()

test_df = pd.read_csv(test_csv_path)
test_students = test_df.sample(n=min(1000, len(test_df)), random_state=1024)
results = {'Random': {'true': [], 'pred': []}, 'DDPG_LSTM_WhiteBox_36D': {'true': [], 'pred': []}}

test_q_col = next((c for c in test_df.columns if 'question' in c.lower() or 'item' in c.lower()), test_df.columns[1])
test_r_col = next((c for c in test_df.columns if 'response' in c.lower() or 'score' in c.lower() or 'correct' in c.lower()), test_df.columns[2])

for strat in ['Random', 'DDPG_LSTM_WhiteBox_36D']:
for _, row in tqdm(test_students.iterrows(), total=len(test_students), desc=f"评估策略: {strat}"):
v_i, v_r = get_clean_sequence(row[test_q_col], row[test_r_col])
if len(v_i) < 16: continue
avail_i, avail_r = v_i.copy(), v_r.copy()
seed_idx = random.choice(range(len(avail_i)))
current_item_id, current_response_val = avail_i.pop(seed_idx), avail_r.pop(seed_idx)
hx, cx = torch.zeros(1, 128, device=device), torch.zeros(1, 128, device=device)
history_i, history_r = [current_item_id], [current_response_val]
for step in range(10):
if not avail_i: break
if strat == 'Random':
idx = random.choice(range(len(avail_i)))
else:
with torch.no_grad():
c_t = torch.tensor([current_item_id], device=device)
ideal_73d_vector, next_hx, next_cx = actor(
frozen_128d_bank[current_item_id].unsqueeze(0), q_masks_tensor[current_item_id].unsqueeze(0),
torch.sigmoid(ncdm_doctor.k_difficulty(c_t)), torch.sigmoid(ncdm_doctor.e_discrimination(c_t)),
torch.tensor([current_response_val], dtype=torch.float, device=device), hx, cx
)
a_t = torch.tensor(avail_i, device=device)
candidate_73d = torch.cat([q_masks_tensor[avail_i], torch.sigmoid(ncdm_doctor.k_difficulty(a_t)), torch.sigmoid(ncdm_doctor.e_discrimination(a_t))], dim=-1)
idx = torch.argmin(torch.cdist(ideal_73d_vector, candidate_73d).squeeze(0)).item()
hx, cx = next_hx, next_cx
next_item_id, next_response_val = avail_i.pop(idx), avail_r.pop(idx)
history_i.append(next_item_id)
history_r.append(next_response_val)
current_item_id, current_response_val = next_item_id, next_response_val

h_t_i, h_t_r = torch.tensor(history_i, dtype=torch.long, device=device), torch.tensor(history_r, dtype=torch.float, device=device)
with torch.no_grad():
init_val = torch.log(torch.sigmoid(ncdm_doctor.k_difficulty(h_t_i)).mean(dim=0).clamp(1e-4, 1-1e-4) / (1.0 - torch.sigmoid(ncdm_doctor.k_difficulty(h_t_i)).mean(dim=0).clamp(1e-4, 1-1e-4)))
alpha = init_val.unsqueeze(0).clone().detach().requires_grad_(True)
opt = optim.Adam([alpha], lr=0.05)
for step_idx in range(8):
opt.zero_grad()
input_x = (torch.sigmoid(ncdm_doctor.e_discrimination(h_t_i))*10) * (torch.sigmoid(alpha) - torch.sigmoid(ncdm_doctor.k_difficulty(h_t_i))) * q_masks_tensor[h_t_i]
input_x = ncdm_doctor.drop_1(torch.sigmoid(ncdm_doctor.prednet_full1(input_x)))
input_x = ncdm_doctor.drop_2(torch.sigmoid(ncdm_doctor.prednet_full2(input_x)))
pred = torch.sigmoid(ncdm_doctor.prednet_full3(input_x)).squeeze(-1)
loss = nn.BCELoss()(pred, h_t_r)
loss.backward()
opt.step()
if step_idx >= 5 and loss.item() < 1e-3: break
with torch.no_grad():
if not avail_i: continue
t_t_i = torch.tensor(avail_i, dtype=torch.long, device=device)
input_x = (torch.sigmoid(ncdm_doctor.e_discrimination(t_t_i))*10) * (torch.sigmoid(alpha) - torch.sigmoid(ncdm_doctor.k_difficulty(t_t_i))) * q_masks_tensor[t_t_i]
input_x = ncdm_doctor.drop_1(torch.sigmoid(ncdm_doctor.prednet_full1(input_x)))
input_x = ncdm_doctor.drop_2(torch.sigmoid(ncdm_doctor.prednet_full2(input_x)))
preds = torch.sigmoid(ncdm_doctor.prednet_full3(input_x)).squeeze(-1)
results[strat]['true'].extend(avail_r)
results[strat]['pred'].extend(preds.cpu().numpy())

print("\n" + "="*65)
print("🏆 【完全体 Colab 轰鸣结束】最终决战战报")
print("="*65)
metrics = []
for strat in ['Random', 'DDPG_LSTM_WhiteBox_36D']:
y_true, y_pred = np.array(results[strat]['true']), np.array(results[strat]['pred'])
auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
acc = accuracy_score(y_true >= 0.5, y_pred >= 0.5)
metrics.append({'Strategy': strat, 'AUC': f"{auc:.4f}", 'Accuracy': f"{acc:.4f}"})
print(pd.DataFrame(metrics).to_markdown(index=False))
print("="*65)







=======================================================
🛡️ Colab 加速防弹版：连续分数 + 智能寻址 + 快速训练
=======================================================
🖥️ 使用设备: cuda
✅ 雷达寻址成功！资产目录: /content/drive/MyDrive/Colab_Projects/KCQRL-main/data/XES3G5M/metadata
✅ 训练数据: /content/drive/MyDrive/Colab_Projects/KCQRL-main/data/XES3G5M/kc_level/train_valid_sequences.csv
✅ 测试数据: /content/drive/MyDrive/test.csv

🔍 自动探测到 CSV 拥有以下列名: ['fold', 'uid', 'questions', 'concepts', 'responses', 'timestamps', 'selectmasks', 'is_repeat']
✅ 已成功智能锁定 -> 题目特征列: [questions], 学生得分列: [responses]
✅ 已成功避开旧模型学生维度冲突，题目判官参数加载完毕！

🔥 显卡已全负荷挂载，开始强化学习矩阵演进...
Epoch 1/5: 100%|██████████| 300/300 [01:50<00:00,  2.72it/s]
✅ Epoch 1 | 均值信息增益: 0.2107 | Critic Loss: 1.1941
Epoch 2/5: 100%|██████████| 300/300 [01:50<00:00,  2.72it/s]
✅ Epoch 2 | 均值信息增益: 0.2278 | Critic Loss: 4.6316
Epoch 3/5: 100%|██████████| 300/300 [01:50<00:00,  2.72it/s]
✅ Epoch 3 | 均值信息增益: 0.1953 | Critic Loss: 19.7473
Epoch 4/5: 100%|██████████| 300/300 [01:50<00:00,  2.72it/s]
✅ Epoch 4 | 均值信息增益: 0.2180 | Critic Loss: 50.6112
Epoch 5/5: 100%|██████████| 300/300 [01:50<00:00,  2.71it/s]
✅ Epoch 5 | 均值信息增益: 0.2323 | Critic Loss: 100.0544

🏆 进入终极盲测推演对决...

















import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================
# 加载已保存的 Actor 模型（无需重新训练）
# ==========================================
actor.load_state_dict(torch.load(f'{base_path}/ddpg_whitebox_36d_actor_colab.pt', map_location=device))
actor.eval()

# ==========================================
# 加载测试数据
# ==========================================
test_path = '/content/drive/MyDrive/Colab_Projects/KCQRL-main/data/XES3G5M/kc_level/test.csv'
test_df = pd.read_csv(test_path)
print(f"✅ 测试数据加载成功，样本数: {len(test_df)}")

# 列名自适应
test_q_col = next((c for c in test_df.columns if 'question' in c.lower() or 'item' in c.lower()), test_df.columns[1])
test_r_col = next((c for c in test_df.columns if 'response' in c.lower() or 'score' in c.lower() or 'correct' in c.lower()), test_df.columns[2])
print(f"🔍 测试集列名 -> 题目: [{test_q_col}], 得分: [{test_r_col}]")

# ==========================================
# 快速推演函数（复用训练时的工具函数）
# ==========================================
def get_clean_sequence(row_q, row_r):
    q_strs = str(row_q).replace('[', '').replace(']', '').replace("'", "").replace('"', '').split(',')
    r_strs = str(row_r).replace('[', '').replace(']', '').replace("'", "").replace('"', '').split(',')
    v_i, v_r = [], []
    for q, r in zip(q_strs, r_strs):
        if not q.strip() or not r.strip(): continue
        q_id = int(float(q.strip()))
        r_val = float(r.strip())
        if 0 <= q_id < MAX_SAFE_ITEM_ID and r_val >= 0.0:
            v_i.append(q_id)
            v_r.append(r_val)
    return v_i, v_r

# 使用训练时定义的 evaluate_entropy_ncdm 函数，但这里我们只需要最终的预测，直接内联简单版本
def predict_remaining(history_i, history_r, avail_i):
    """根据历史答题记录，预测剩余题目的答对概率"""
    if len(history_i) == 0:
        alpha = torch.zeros((1, K_DIM), device=device, requires_grad=True)
    else:
        h_t_i = torch.tensor(history_i, dtype=torch.long, device=device)
        h_t_r = torch.tensor(history_r, dtype=torch.float, device=device)
        with torch.no_grad():
            hist_diffs = torch.sigmoid(ncdm_doctor.k_difficulty(h_t_i))
            mean_diff = hist_diffs.mean(dim=0).clamp(1e-4, 1 - 1e-4)
            init_val = torch.log(mean_diff / (1.0 - mean_diff))
        alpha = init_val.unsqueeze(0).clone().detach().requires_grad_(True)
    
    opt = optim.Adam([alpha], lr=0.05)
    if len(history_i) > 0:
        for step_idx in range(8):
            opt.zero_grad()
            diff = torch.sigmoid(ncdm_doctor.k_difficulty(h_t_i))
            disc = torch.sigmoid(ncdm_doctor.e_discrimination(h_t_i)) * 10
            input_x = disc * (torch.sigmoid(alpha) - diff) * q_masks_tensor[h_t_i]
            input_x = ncdm_doctor.drop_1(torch.sigmoid(ncdm_doctor.prednet_full1(input_x)))
            input_x = ncdm_doctor.drop_2(torch.sigmoid(ncdm_doctor.prednet_full2(input_x)))
            pred = torch.sigmoid(ncdm_doctor.prednet_full3(input_x)).squeeze(-1)
            loss = nn.BCELoss()(pred, h_t_r)
            loss.backward()
            opt.step()
            if step_idx >= 5 and loss.item() < 1e-3:
                break
    
    with torch.no_grad():
        t_t_i = torch.tensor(avail_i, dtype=torch.long, device=device)
        diff = torch.sigmoid(ncdm_doctor.k_difficulty(t_t_i))
        disc = torch.sigmoid(ncdm_doctor.e_discrimination(t_t_i)) * 10
        input_x = disc * (torch.sigmoid(alpha) - diff) * q_masks_tensor[t_t_i]
        input_x = ncdm_doctor.drop_1(torch.sigmoid(ncdm_doctor.prednet_full1(input_x)))
        input_x = ncdm_doctor.drop_2(torch.sigmoid(ncdm_doctor.prednet_full2(input_x)))
        preds = torch.sigmoid(ncdm_doctor.prednet_full3(input_x)).squeeze(-1)
    return preds.cpu().numpy()

# ==========================================
# 开始推演
# ==========================================
test_students = test_df.sample(n=min(1000, len(test_df)), random_state=1024)
results = {'Random': {'true': [], 'pred': []}, 'DDPG_LSTM_WhiteBox_36D': {'true': [], 'pred': []}}

for strat in ['Random', 'DDPG_LSTM_WhiteBox_36D']:
    for _, row in tqdm(test_students.iterrows(), total=len(test_students), desc=f"评估策略: {strat}"):
        v_i, v_r = get_clean_sequence(row[test_q_col], row[test_r_col])
        if len(v_i) < 16: continue
        avail_i, avail_r = v_i.copy(), v_r.copy()
        seed_idx = random.choice(range(len(avail_i)))
        current_item_id, current_response_val = avail_i.pop(seed_idx), avail_r.pop(seed_idx)
        hx, cx = torch.zeros(1, 128, device=device), torch.zeros(1, 128, device=device)
        history_i, history_r = [current_item_id], [current_response_val]
        
        for step in range(10):
            if not avail_i: break
            if strat == 'Random':
                idx = random.choice(range(len(avail_i)))
            else:
                with torch.no_grad():
                    c_t = torch.tensor([current_item_id], device=device)
                    ideal_73d_vector, next_hx, next_cx = actor(
                        frozen_128d_bank[current_item_id].unsqueeze(0), q_masks_tensor[current_item_id].unsqueeze(0),
                        torch.sigmoid(ncdm_doctor.k_difficulty(c_t)), torch.sigmoid(ncdm_doctor.e_discrimination(c_t)),
                        torch.tensor([current_response_val], dtype=torch.float, device=device), hx, cx
                    )
                a_t = torch.tensor(avail_i, device=device)
                candidate_73d = torch.cat([q_masks_tensor[avail_i], torch.sigmoid(ncdm_doctor.k_difficulty(a_t)), torch.sigmoid(ncdm_doctor.e_discrimination(a_t))], dim=-1)
                idx = torch.argmin(torch.cdist(ideal_73d_vector, candidate_73d).squeeze(0)).item()
                hx, cx = next_hx, next_cx
                
            next_item_id, next_response_val = avail_i.pop(idx), avail_r.pop(idx)
            history_i.append(next_item_id)
            history_r.append(next_response_val)
            current_item_id, current_response_val = next_item_id, next_response_val

        # 对剩余题目进行预测
        preds = predict_remaining(history_i, history_r, avail_i)
        results[strat]['true'].extend(avail_r)
        results[strat]['pred'].extend(preds)

# ==========================================
# 计算并输出最终战报
# ==========================================
print("\n" + "="*65)
print("🏆 【Colab 快速实验】最终决战战报")
print("="*65)
metrics = []
for strat in ['Random', 'DDPG_LSTM_WhiteBox_36D']:
    y_true = np.array(results[strat]['true'])
    y_pred = np.array(results[strat]['pred'])
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true >= 0.5, y_pred >= 0.5)
    metrics.append({'Strategy': strat, 'AUC': f"{auc:.4f}", 'Accuracy': f"{acc:.4f}"})
print(pd.DataFrame(metrics).to_markdown(index=False))
print("="*65)  ✅ 测试数据加载成功，样本数: 3613
🔍 测试集列名 -> 题目: [questions], 得分: [responses]评估策略: Random: 100%|██████████| 1000/1000 [00:14<00:00, 68.36it/s]
评估策略: DDPG_LSTM_WhiteBox_36D: 100%|██████████| 1000/1000 [00:27<00:00, 36.63it/s]=================================================================
🏆 【Colab 快速实验】最终决战战报
=================================================================
| Strategy               |   AUC |   Accuracy |
|:-----------------------|------:|-----------:|
| Random                 | 0.723 |     0.7958 |
| DDPG_LSTM_WhiteBox_36D | 0.736 |     0.8047 |  这个似乎是唯一较成功的代码
