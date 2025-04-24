# STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM

## Abstract
Transformer-based models are the foundation of modern machine learning, but their execution, particularly during autoregressive decoding in large language models (LLMs), imposes significant strain on the memory system due to frequent memory accesses and growing key-value (KV) caches. This creates a bottleneck in memory bandwidth, especially as context lengths increase. Processing-in-memory (PIM) architectures are a promising solution, offering high internal bandwidth and compute parallelism near memory. However, current PIM designs are primarily optimized for dense attention and struggle with the dynamic, irregular access patterns introduced by modern KV cache sparsity techniques. Consequently, they suffer from workload imbalance, reducing throughput and resource utilization. In this work, we propose STARC, a novel sparsity-optimized data mapping scheme tailored specifically for efficient LLM decoding on PIM architectures. STARC clusters KV pairs based on semantic similarity and maps them to contiguous memory regions aligned with PIM bank structures. During decoding, queries retrieve relevant tokens at cluster granularity by matching against precomputed centroids, enabling selective attention and parallel processing without frequent reclustering or data movement overhead. Experiments on the HBM-PIM system show that, compared to common token-wise sparsity methods, STARC reduces attention-layer latency by 19%–31% and energy consumption by 19%–27%. Under a KV cache budget of 1024, it achieves up to 54%–74% latency reduction and 45%–67% energy reduction compared to full KV cache retrieval. Meanwhile, STARC achieves model accuracy comparable to state-of-the-art sparse attention methods, demonstrating its effectiveness in enabling efficient and hardware-friendly long-context LLM inference on PIM architectures.


## Installation
1. Clone this repo (also clone submodules)
```
git clone --recurse-submodules https://github.com/EPIC-RPI/STARC
cd STARC
```
2. Install dependency libraries
```
conda create -yn quest python=3.10
conda activate quest
pip install ninja==1.11.1.1 packaging
pip install -e . && pip install flash-attn==2.3.0 --no-build-isolation
conda install -c rapidsai -c nvidia -c conda-forge cuml
conda install -c conda-forge cupy
conda install numpy scikit-learn
conda install datasets

# Install CMake (with version >= 3.26.4)
conda install cmake
```


## Accuracy Evaluation
Our evaluations are based on [LongChat-7B-v1.5-32K](https://huggingface.co/lmsys/longchat-7b-v1.5-32k?clone=true) model, which is capable of handling long-context text generations. We evaluate the LongBench benchmarks and provide several scripts to reproduce the results presented in our paper.

To reproduce the LongBench results, please modify and execute:
```
bash scripts/longbench.sh
```

To evaluate the perplexity result of PG-19, please execute:
```
bash scripts/ppl_eval.sh
```