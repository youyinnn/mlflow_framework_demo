import time
import GPUtil
import psutil
import numpy as np

from pathlib import Path
import os
import requests
import socket

def get_gpu_usage():
    # 获取 GPU 信息
    gpus = GPUtil.getGPUs()
    gpu_usage = []
    
    for gpu in gpus:
        gpu_usage.append({
            'id': gpu.id,
            'name': gpu.name,
            'memoryTotal': gpu.memoryTotal,
            'memoryFree': gpu.memoryFree,
            'memoryUsed': gpu.memoryUsed,
            'load': gpu.load * 100  # 转换为百分比
        })
    
    return gpu_usage

def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

def send_msg(msg):
    print(msg)
    headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjI5NzM1MSwidXVpZCI6IjA4YWM4MTQ0LWY1ZDUtNDBiNC05ZjY2LTQ5ZWRlMGFhZTRjZiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.ZRg1XZ7ApeaiU4mrNbeUST7BK3VVkiWv1fG26DLktlKRa4yUB9fPV5a-YJfFtnXYyJ7ccjqn-xkr60EuzKFYPQ".strip()}
    resp = requests.post(
        "https://www.autodl.com/api/v1/wechat/message/send",
        json={
            "title": "",
            "name": msg,
            "content": ""
        },
        headers=headers,
    )
    print(resp.content)

if __name__ == '__main__':
    print('start mo')
    hostname_postfix = socket.gethostname().split('-')[-1]

    # the last minute
    len_lim = 60

    # low usage in five minutes
    warining_thresholds = np.array([1,3,5,7]) * 60
    shutdown_threshold = 60 * 10

    avg = np.array([])
    count = 0
    print_count = 0
    while True:
        time.sleep(1)
        gpu_usage = get_gpu_usage()
        for gpu in gpu_usage:
            if len(avg) == len_lim:
                avg = avg[1:]
            avg = np.append(avg, gpu['load'])
        if avg.mean() < 5:
            count  += 1
        else:
            count = 0
        print_count += 1
        if print_count == 30:
            print(avg.mean(), count)
            print_count = 0
        for i, wt in enumerate(warining_thresholds):
            if count == wt:
                send_msg(f"GPU load usage, {hostname_postfix}. Shutdown warning of #{i}.")
            
        
        if count == shutdown_threshold:
            send_msg(f"GPU load usage, {hostname_postfix}. Shuting down.")
            os.system("/usr/bin/shutdown")