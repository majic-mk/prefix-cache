"""Small real CUDA smoke only, not vLLM/cache/model qualification."""
import json, subprocess
import torch

inventory=subprocess.run(["nvidia-smi","-i","GPU-8b500efe-1a50-0e8e-b21e-716807eebedf",
    "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu,driver_version",
    "--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip()
assert torch.cuda.device_count()==1
free,total=torch.cuda.mem_get_info(0)
assert free>1<<30, "insufficient free memory for bounded smoke"
a=torch.arange(1024,dtype=torch.float32,device="cuda")
b=(a*2+1).cpu()
torch.cuda.synchronize()
assert torch.equal(b,torch.arange(1024,dtype=torch.float32)*2+1)
print(json.dumps({"inventory":inventory,"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
    "device_name":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0),
    "free_bytes_before":free,"total_bytes":total,"tensor_elements":1024,"result_exact":True,
    "real_cuda_kernel_executed":True,"vllm_verified":False,"cache_verified":False,"model_download_bytes":0},indent=2))
