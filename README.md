# ArgRAG:用于MMKG-RAG的多智能体路径推理
<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-custom-lightgrey">
</p>

## 🚀 Overview
## 基线
**MegaRAG-Multimodal Graph-based Retrieval Augmented Generation**
### 背景
在 claim verification 上证明了，两个持对立立场的 agent 进行多轮辩论，再由 moderator 裁决，能比非辩论方法得到更高的准确率和更好的 justification，原因就在于对抗机制更容易暴露单 agent 的忽略、误读和过度猜测。
当前大多数基于多智能体的RAG 都是协作分工，比如将问题拆解、并行检索，然后生成答案（HM-RAG）；或者针对不同模态协作生成（MeagRAG）。然而在语义上相似的内容，并不总是提供支持证据，通常是支持证据与干扰证据混杂，并且不同的模态可能给出信息是有冲突的。
基于 MegaRAG 方法：你主要需要改动在对检索部分的优化，不需要优化构图部分。
  - 支持证据与干扰证据混杂，相关不等于可回答
  - 模态冲突，文本、图像、表格可能给出不一致信号
  - 图谱连接错误，视觉实体对齐错了会把整条推理链带偏
### 检索部分的优化设计
我们提出多智能体博弈，针对检索的内容，使用持不同立场的智能体进行辩论，然后再由裁决者根据整个辩论过程，得出最终能够支持回答的一致证据。
辩论触发器：在初始召回结果中，很多内容只是语义上相近，但是否真正构成支持证据，取决于它是否被周围结构充分支撑。一条边如果能被多条短路径、邻域重合、规则模式反复支撑，它更像“稳固事实”；反之，如果它是孤立边或缺乏局部支撑，它更像干扰项或脆弱证据。
  $$StructSup(e)=w_1N_{path}(h,t)+w_2N_{common}(h,t)$$
1. 路径支撑 $$N_{path}(h,t)$$ 对候选三元组 $$e=(h,r,t)$$ 只在以 $$h,t$$ 为端点的局部子图中搜索长度为 2 或 3 的简单路径 $$\pi:h \xrightarrow[]{r1} x \xrightarrow[]{r2} t$$ 或  $$\pi:h \xrightarrow[]{r1} x \xrightarrow[]{r2} y \xrightarrow[]{r3} t $$
2. 共同邻居数量 $$N_{common}(h,t)$$
触发分数进行归一化，并设置阈值，低于阈值 $$\tau$$ 的三元组触发辩论。
### 多智能体设计
- 支持智能体：从证据池里挑出最可能支撑问题求解的三元组、路径、页面区域，识别最能支撑当前问题求解的证据，并给出保留理由。
- 反驳智能体：指出哪些边、路径、页面虽然相关，但其实不支持回答，或者可能导向错误解释。
- 歧义智能体：识别模态歧义，比如图像区域不清、表格列头含糊、实体对齐不稳。
- 结构核验智能体：任务是检查当前证明子图里是否存在不可信的图边，如跨模态实体错链、关系方向错误、下一跳邻居引入了无关路径等。
- 裁决智能体：根据整个辩论过程，得出最终的三元组集合。
- 稳健答案生成：使用最终可信子图以及辩论摘要进行答案生成。

## 📦 Installation
#### Requirements

* Python 3.9+ (3.10 recommended)
* GPU for embeddings inference
* OpenAI API key

### Step 1: Install [MinerU](https://github.com/opendatalab/MinerU/tree/release-1.3.6)

```bash
# git clone -b release-1.3.6 https://github.com/opendatalab/MinerU.git
git clone https://github.com/opendatalab/MinerU.git
cd MinerU

conda create --name mineru python=3.10 -y
conda activate mineru
pip install -e .

pip install huggingface_hub accelerate accelerate
wget https://raw.githubusercontent.com/opendatalab/MinerU/refs/heads/release-1.3.6/scripts/download_models_hf.py -O download_models_hf.py

sed -i "s|https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json|https://raw.githubusercontent.com/opendatalab/MinerU/release-1.3.6/magic-pdf.template.json|" download_models_hf.py

python download_models_hf.py
```

### Step 2: Install MegaRAG

```bash
git clone https://github.com/AI-Application-and-Integration-Lab/MegaRAG.git
cd MegaRAG

# Install missing packages
conda activate mineru
pip install -r requirements_mineru.txt

conda create --name megarag python=3.10 -y
conda activate megarag
pip install -e .

# Fill in your OPENAI_API_KEY and MINERU_PATH in env.sh
cp .env.sh env.sh

# Install lightRAG
mkdir lib && cd lib
git clone --branch v1.4.3 https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .
```

## ⚡ Quickstart

### 1. Use the Tiny Example

```bash
cd egs/world_history_tiny
mkdir data
```

### 2. Download Example PDF

Download the example PDF, and query file from [Google Drive](https://drive.google.com/drive/folders/1iuukUWsxMYobuDRLRJ3dBOkB9mdPGoPp?usp=sharing) and place it in the `data/` folder.

### 3. Build the Multimodal Knowledge Graph

```bash
bash ./run_build_mmkg.sh
```

### 4. Query with MegaRAG

```bash
bash ./run_querying.sh
```

## 📂 Using Your Own Dataset

### 1. Create a New Recipe

```bash
cp -r egs/.template egs/<your_dataset>
# Change <your_dataset> to your dataset name
cd egs/<your_dataset>
mkdir data
```

### 2. Add Your Data

Place your documents (PDF) in the `data/` folder.

### 3. Edit the Config File

Modify the entity types or other settings in `conf/addon_params.yaml` to match your data.

### 4. Build and Query

```bash
bash egs/<your_dataset>/run_build_mmkg.sh
# MegaRAG
bash egs/<your_dataset>/run_querying.sh
# AargRAG
bash egs/<your_dataset>/run_querying_mmkg_debate.sh
# Make sure to update filenames in these scripts accordingly.
```

## Acknowledgments

AargRAG is inspired by the work of [MegaRAG](https://github.com/AI-Application-and-Integration-Lab/MegaRAG). We are grateful for their excellent tools and contributions.

## 📄 License

This project is released under a custom license.
See the [LICENSE](./LICENSE) file for full terms and conditions.

For academic or commercial use, please contact the authors directly.
