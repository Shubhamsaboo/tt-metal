# lm-eval-harness.py

To run zero-shot evaluations of our model (to compare against against to Table
3 of the Mamba paper), we use the lm-evaluation-harness library.

To install the `lm-evaluation-harness` library, run the following commands:

```sh
# Clone and then install as an editable package
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# Upgrade datasets package to fix an issue when downloading datasets
pip3 install datasets==2.16.0
```

## CPU

Running against the CPU reference model (`MambaDecode`)

```sh
python benchmarks/lm_harness_eval.py --tasks hellaswag \
    --device cpu --batch_size 16 --limit 0.1 \
    --model mamba-cpu-reference
```
