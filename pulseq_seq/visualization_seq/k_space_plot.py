from pathlib import Path
from pypulseq.Sequence.sequence import Sequence
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

seq_path = Path("/home/universe/SVR/real_time_seq_usc/pulseq_seq/out_seq/uih790_rtspiral_realtime.seq")
traj_path = Path("/home/universe/SVR/real_time_seq_usc/pulseq_seq/out_trajectory/fa2f521bc41db9ffa7fea48d376fafa7.mat")

seq = Sequence()
seq.read(str(seq_path))

k_adc, *_ = seq.calculate_kspace()
traj = loadmat(traj_path)

base_k = traj["base_k_played"]   # [Nsamples, 2]
global_angles = traj["global_arm_angle_deg"].ravel()

# 画 base arm
plt.figure()
plt.plot(base_k[:,0], base_k[:,1])
plt.axis("equal")
plt.title("Base played arm")
plt.show()