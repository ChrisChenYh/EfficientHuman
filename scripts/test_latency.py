import torch
from litehuman.models import SMPLCamRLESimccVM
from litehuman.opt import cfg
from litehuman.models import builder
from torchsummary import summary
import time
from thop import clever_format, profile

model = builder.build_sppe(cfg.MODEL)
model.eval()
inps=torch.zeros((1,3,256,256), dtype=torch.float32).to('cpu')

flops, params = profile(model, inputs=(inps,))
macs, params = clever_format([flops, params], "%.3f")
print(flops)
print(params)
print(macs)
start_time = time.time()
time_sum = 0
for i in range(1000):
    outs = model(inps)
    expand_time = time.time() - start_time
    print(expand_time)
    start_time = time.time()
    time_sum += expand_time
print(time_sum)

