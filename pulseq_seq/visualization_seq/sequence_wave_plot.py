from pathlib import Path
from pypulseq.Sequence.sequence import Sequence
import matplotlib.pyplot as plt

seq_path = Path("/home/universe/SVR/real_time_seq_usc/pulseq_seq/out_seq/uih790_rtspiral_realtime.seq")

seq = Sequence()
seq.read(str(seq_path))

# 画前一小段时间，先观察局部波形
seq.plot(time_range=(0, 0.07))   # 前 70 ms
plt.show()