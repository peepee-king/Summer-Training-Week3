import json
import matplotlib.pyplot as plt
import os
import numpy as np
file_path="resnet_24000_city/metrics.json"

def read_loss_curve(metrics_path):
	iters, losses = [], []
	aps, aps_iters = [], []
	with open(metrics_path, "r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			if "iteration" in obj and "total_loss" in obj:
				iters.append(int(obj["iteration"]))
				losses.append(float(obj["total_loss"]))
			if "bbox/AP" in obj and "iteration" in obj:
				aps.append(float(obj["bbox/AP"]))
				aps_iters.append(int(obj["iteration"]))
	# 確保按 iter 排序
	order_loss = np.argsort(iters)
	iters = list(np.array(iters)[order_loss])
	losses = list(np.array(losses)[order_loss])
	if len(aps_iters) > 0:
		order_ap = np.argsort(aps_iters)
		aps_iters = list(np.array(aps_iters)[order_ap])
		aps = list(np.array(aps)[order_ap])
	return iters, losses, aps, aps_iters

iters, losses, aps, aps_iters = read_loss_curve(file_path)
os.makedirs("./loss_AP_img", exist_ok=True)

plt.figure(figsize=(9,4.5))
plt.plot(iters, losses, label="total_loss", color="#1f77b4", linewidth=1)
plt.xlabel("iteration")
plt.ylabel("total_loss")
plt.title("Total Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"./loss_AP_img/{file_path.split('/')[0]}_loss.png", dpi=150)

plt.figure(figsize=(9,4.5))
plt.plot(aps_iters, aps, label="bbox/AP", color="#ff7f0e", linewidth=1)
plt.xlabel("iteration")
plt.ylabel("AP")
plt.title("AP")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(f"./loss_AP_img/{file_path.split('/')[0]}_AP.png", dpi=150)
print(f"已儲存圖檔: {file_path.split('/')[0]}.png")