from datetime import datetime

import torch as t
import argparse
from timeit import default_timer as timer

import torch.cuda
# from datetime import datetime
from diffusers import StableDiffusionPipeline

import sys

prog = 'corpusdiffuse'
__version__ = "0.0.1"

# Mask out if uploaded to GitHub
# hf_token="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"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--version", action='version', version='%(prog)s ' + __version__)
    parser.add_argument("--src", help="source file")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--stats", help="runtime stats report")
    parser.add_argument("--start", type=int, help="start at line #")
    parser.add_argument("--end", type=int, help="end at line #")
    parser.add_argument("--tsv", default=False, action="store_true", help="is input TSV? (default=False)")

    args = parser.parse_args()

    # get your token at https://huggingface.co/settings/tokens
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)

    # hax
    pipe.safety_checker = lambda images, **kwargs: (images, False)

    # maybe check if cuda is available
    pipe.to("cuda")

    elapsed = 0.0
    ctr = 1.0

    with open(args.src, 'r') as src, open(args.stats, 'w') as stats:

        stats.write(f"Started at {datetime.now()}\n")
        stats.write(f"Cuda device: {torch.cuda.get_device_name()}\n")
        stats.write(f"Cuda mem: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory}\n")
        stats.write(f"Line start: {args.start}")
        stats.write(f"Line end: {args.end}")

        starttime = timer()

        for i, line in enumerate(src, start=1):

            if i >= args.start and i <= args.end:
                line = line.strip()

                if args.tsv:
                    line = line.split('\t')[1]

                cur_start = timer()
                image = pipe(line)["sample"][0]
                image.save(f"{args.out}/{i}.png")
                cur_end = timer()
                cur_elapsed = cur_end - cur_start

                print(f"{i}\t {cur_elapsed:0.4f} sec \t{line[:20]}")
                stats.write(f"{i}\t {cur_elapsed:0.4f} sec \t{line[:20]}\n")

            ctr += 1

        endtime = timer()

        print(f"Elapsed time: {endtime - starttime:0.4f} sec.")
        stats.write(f"Elapsed time: {endtime - starttime:0.4f} sec.\n")

        # foo
