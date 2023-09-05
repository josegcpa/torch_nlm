import time
import argparse
import torch
from torch_nlm import nlm2d, nlm3d
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_size",default=[128,128],type=int,nargs="+")
    parser.add_argument("--kernel_size",default=11,type=int)
    parser.add_argument("--std",default=1.0,type=float)
    parser.add_argument("--sub_filter_size",default=32,type=int)
    parser.add_argument("--dev",default="cpu",type=str)
    parser.add_argument("--n",type=int,default=100)

    args = parser.parse_args()

    dev = "cuda:1"

    if len(args.image_size) == 2:
        fn = nlm2d
    elif len(args.image_size) == 3:
        fn = nlm3d

    times = []
    for i in trange(args.n):
        image = torch.rand(*args.image_size).to(args.dev)

        a = time.time()
        output = fn(
            image,
            kernel_size=args.kernel_size,
            std=args.std,
            sub_filter_size=args.sub_filter_size)
        b = time.time()

        if i > 0:
            times.append(b-a)

    print("Average time:", sum(times)/len(times))