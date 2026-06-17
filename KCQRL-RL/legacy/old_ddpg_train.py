import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from google.colab import drive
print("=======================================================")
print("💎 终极完全体：36维专家级 Q矩阵 + 官方 NCDM 联合特训")
print("=======================================================")
# 1. 挂载云盘与环境配置
if not os.path.exists('/content/drive'):
drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 当前使用的计算设备: {device}")

base_path = '/content/drive/MyDrive/Colab_Projects/KCQRL-main/data/XES3G5M'
metadata_path = f'{base_path}/metadata'
# 您提供的绝对正确的数据路径！
real_data_path = f'{base_path}/real_data/XES3G5M/question_level/test_quelevel.csv'
# ⚠️ 核心替换 1：加载我们刚刚铸造的 36 维专家矩阵！
multi_hot_q_path = f'{metadata_path}/q_matrix_multihot_36_expert.pt'
# ⚠️ 核心替换 2：保存名为 36d_expert_best，防止与您的 64d 模型冲突
ncdm_36d_save_path = f'{metadata_path}/ncdm_model_36d_expert_best.pt'
# 2. 加载 36 维多热 Q矩阵
q_masks_tensor = torch.load(multi_hot_q_path).to(device)
# ⚠️ 核心替换 3：维度改为 36！
K_DIM = 36 
num_items = q_masks_tensor.shape[0]
print(f"✅ 成功加载 {K_DIM} 维 Q 矩阵，题目总数: {num_items}")
# 3. 核心修复：建立连续且安全的用户索引 (防止 CUDA 越界)
print(f"⏳ 正在读取交互数据: {os.path.basename(real_data_path)}...")
df_ncdm = pd.read_csv(real_data_path)
unique_uids = df_ncdm['uid'].unique()
uid_to_idx = {uid: idx for idx, uid in enumerate(unique_uids)}
num_users = len(uid_to_idx)
print(f"👥 成功建立用户映射字典，共 {num_users} 名学生。")
# 4. 严格复刻官方的模型结构与正值裁剪器 (Clipper)
class NoneNegClipper(object):
def __init__(self):
super(NoneNegClipper, self).__init__()
def __call__(self, module):
if hasattr(module, 'weight'):
w = module.weight.data
a = torch.relu(torch.neg(w))
w.add_(a)
class OfficialNCDM(nn.Module):
def __init__(self, student_n, exer_n, knowledge_n):
super(OfficialNCDM, self).__init__()
self.knowledge_dim = knowledge_n
self.exer_n = exer_n
self.emb_num = student_n
self.prednet_len1, self.prednet_len2 = 512, 256 
self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
self.e_discrimination = nn.Embedding(self.exer_n, 1)
self.prednet_full1 = nn.Linear(self.knowledge_dim, self.prednet_len1)
self.drop_1 = nn.Dropout(p=0.5)
self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
self.drop_2 = nn.Dropout(p=0.5)
self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
for name, param in self.named_parameters():
if 'weight' in name:
nn.init.xavier_normal_(param)
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
optimizer = optim.Adam(ncdm_doctor.parameters(), lr=0.002)
criterion = nn.BCELoss()
# 5. 数据处理与构建训练样本
print("🔄 正在构建训练样本...")
train_pairs = []
for _, row in df_ncdm.iterrows():
raw_uid = int(row['uid'])
safe_uid = uid_to_idx[raw_uid]
qids = [int(float(q.strip())) for q in str(row['questions']).split(',') if q.strip()]
resps = [float(r.strip()) for r in str(row['responses']).split(',') if r.strip()]
for q, r in zip(qids, resps):
if q < num_items:
train_pairs.append((safe_uid, q, r))
np.random.seed(42) # 固定种子，保证验证集划分一致，方便比较
np.random.shuffle(train_pairs)
split_idx = int(len(train_pairs) * 0.8)
train_data, valid_data = train_pairs[:split_idx], train_pairs[split_idx:]
print(f"✅ 构建完成！训练集: {len(train_data)} 样本 | 验证集: {len(valid_data)} 样本")
# 6. 训练流程：加入“最佳模型保存”逻辑
EPOCHS = 5
BATCH_SIZE = 256
best_auc = 0.0
print("🚀 开始 36 维专属特训，将自动为您保留巅峰时刻的模型...")
for epoch in range(EPOCHS):
ncdm_doctor.train()
total_loss = 0
np.random.shuffle(train_data)
for i in tqdm(range(0, len(train_data), BATCH_SIZE), desc=f"Epoch {epoch+1} 训练"):
batch = train_data[i:i+BATCH_SIZE]
u_batch = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
q_batch = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
r_batch = torch.tensor([x[2] for x in batch], dtype=torch.float, device=device)
optimizer.zero_grad()
preds = ncdm_doctor(u_batch, q_batch, q_masks_tensor[q_batch])
loss = criterion(preds, r_batch)
loss.backward()
optimizer.step()
# 强制裁剪负数权重
ncdm_doctor.apply_clipper()
total_loss += loss.item()
ncdm_doctor.eval()
val_preds, val_trues = [], []
with torch.no_grad():
for i in range(0, len(valid_data), BATCH_SIZE):
batch = valid_data[i:i+BATCH_SIZE]
u_batch = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
q_batch = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
r_batch = [x[2] for x in batch]
preds = ncdm_doctor(u_batch, q_batch, q_masks_tensor[q_batch]).cpu().numpy()
val_preds.extend(preds)
val_trues.extend(r_batch)
val_auc = roc_auc_score(val_trues, val_preds)
val_acc = accuracy_score(val_trues, (np.array(val_preds) > 0.5).astype(int))
avg_loss = total_loss / (len(train_data) / BATCH_SIZE)
# 🏆 核心逻辑：自动保存最佳模型
if val_auc > best_auc:
best_auc = val_auc
torch.save(ncdm_doctor.state_dict(), ncdm_36d_save_path)
print(f"🌟 发现最佳模型！Epoch {epoch+1} | Loss: {avg_loss:.4f} | 验证集 AUC: {val_auc:.4f} | ACC: {val_acc:.4f} | 已覆盖保存！")
else:
print(f"📊 Epoch {epoch+1} | Loss: {avg_loss:.4f} | 验证集 AUC: {val_auc:.4f} | ACC: {val_acc:.4f} (未超越历史最高)")
print(f"\n🎉 36维特训结束！最终保留的最强判官模型 AUC 为: {best_auc:.4f}")
print(f"🏆 巅峰模型已安全存入: {ncdm_36d_save_path}")


