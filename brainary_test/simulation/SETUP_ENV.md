# Simulation 物理沙盒 —— 环境配置

物理沙盒(sapien + GroundingDINO + SAM + pytorch3d/nvdiffrast + ConceptGraphs/FoundationPose)是一套
**重型 CUDA 环境**,与 `env_isaaclab` 依赖互斥,**必须单独建一个 conda 环境**(建议名 `brainary_sim`)。

> ✅ **本机已实测:环境全装好、`PhysicalValidator` 构造成功(GDINO+SAM 模型加载、沙盒就绪)**。
> 注:`nvidia-smi`(用户态工具)会报 "Driver/library version mismatch",但这**只是那个命令行工具的版本问题**——
> CUDA 运行时/驱动本身正常:`torch.cuda.is_available()=True`,可见 2× Quadro RTX 8000。所以 GPU 可用,不必等修驱动。

## 1. 建环境 + 装 torch(CUDA 12.1)

```bash
conda create -y -n brainary_sim python=3.10
conda activate brainary_sim
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# conda 环境【自带】CUDA 12.1 toolkit(独立于系统 CUDA,供后面编译 pytorch3d/GDINO/nvdiffrast):
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit
# 若报 md5 不匹配 -> `conda clean --tarballs --index-cache -y` 后重试(缓存损坏所致)。
nvcc --version   # 应显示 release 12.1
```

## 2. 核心物理/视觉依赖(pip wheel,无需编译)

```bash
pip install sapien==3.0.1 open3d==0.19.0 opencv-python==4.11.0.86 numpy==1.26.4 scipy==1.13.1 \
            trimesh transforms3d pyquaternion supervision==0.28.0 timm==1.0.27 \
            transformers==4.36.0 huggingface_hub pyyaml Pillow requests
pip install "setuptools<81"   # ⚠️ sapien 依赖 pkg_resources;setuptools>=81 已移除它,否则 import sapien 报错
```

> **本机 `brainary_sim` 现状:全部装好并验证 ✅**。torch 2.1.2+cu121 + conda 自带 CUDA 12.1 toolkit(nvcc 12.1.105)+
> GroundingDINO(CUDA 扩展 `_C` 已编译)+ segment_anything + pytorch3d 0.7.9(源码编译)+ nvdiffrast 0.4.0 +
> ConceptGraphs + sapien 3.0.1 都可 import,`PhysicalValidator('config/global_config.yaml')` 构造成功。
> 关键:conda 环境**自带** CUDA 12.1 toolkit,独立于系统 CUDA 13,pytorch3d/nvdiffrast 就对着它编译——系统 CUDA 版本无所谓。

## 3. 需从源码/本地装的(requirements.txt 里的 file:/// 和 git+ 项)

这些在 requirements.txt 里指向对方机器路径,本仓库已把源码放在
`simulation/src/perception/third_party/`,改为从本地 editable 装:

先设编译要用的环境变量 + 装编译辅助包 + 几个漏网 pip 依赖:
```bash
export CUDA_HOME=/home1/banghai/miniconda3/envs/brainary_sim   # 指向本 conda 环境(自带 nvcc 12.1)
export PATH=$CUDA_HOME/bin:$PATH
pip install ninja wheel openai psutil transformations distinctipy ruamel.yaml pyglet \
            mplib==0.2.1 warp-lang pyrender pycollada toppra
```
**关键:所有源码编译都要加 `--no-build-isolation`**(否则 pip 隔离构建环境里没 torch,setup.py `import torch` 会失败):
```bash
cd simulation/src/perception/third_party
pip install -e Grounded-Segment-Anything/GroundingDINO   --no-build-isolation   # 编 _C CUDA 扩展
pip install -e Grounded-Segment-Anything/segment_anything                        # 纯 python
pip install -e ConceptGraphs                              --no-build-isolation
pip install "git+https://github.com/NVlabs/nvdiffrast.git@253ac4f"        --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git@b73d735" --no-build-isolation  # 最慢,~15min
```
> ✅ 已验证:因为 conda 环境**自带 CUDA 12.1 toolkit**(见第 0 步说明),编译都对着它做,系统 CUDA 13 不影响。
> import 时先 `import torch` 再 `from groundingdino import _C`(否则报 libc10.so 找不到)。

### 3.1 gradslam / chamferdist(requirements.txt 里的另外两个 file:/// 项)

`requirements.txt` 里这两项指向原作者机器 `/home/zhizhen/PycharmProjects/simulation/...`，本仓库原先没带源码。
按 ConceptGraphs 官方 README 的要求克隆安装(gradslam 必须切 `conceptfusion` 分支)，已放在
`src/perception/third_party/` 下：

```bash
cd simulation/src/perception/third_party
git clone https://github.com/krrish94/chamferdist.git
git clone https://github.com/gradslam/gradslam.git && (cd gradslam && git checkout conceptfusion)

export CUDA_HOME=$CONDA_PREFIX && export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="7.5"          # Quadro RTX 8000 = sm_75
pip install -e chamferdist --no-build-isolation   # 编 CUDA 扩展
pip install -e gradslam   --no-build-isolation
```
> 注：主管线(`PerceptionPipeline` -> `cg_wrapper`)只用 GroundingDINO + SAM，**不 import gradslam/chamferdist**；
> 它们只被 ConceptGraphs 自己的 SLAM 脚本用到。装上是为了对齐 requirements.txt。

### 3.2 requirements.txt 里漏装的普通 pip 包

```bash
pip install Cython==3.2.5 docstring_parser==0.18.0 eval_type_backport==0.3.1 \
            importlib_resources==6.5.2 pybind11==3.0.4 PyOpenGL-accelerate==3.1.10 \
            typeguard==4.5.2 tyro==1.0.13
pip install --no-deps torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```
> `--no-deps` 是为了防止 torchaudio 顺手把 torch 重装/降级。装完 `pip check` 无冲突。
>
> ⚠️ `requirements.txt` 未改动(仍是原作者机器的 `file:///home/zhizhen/...` 路径)，所以
> **不能直接 `pip install -r requirements.txt`**，按本文档分步装。另外该文件里几十个包的 pin
> 版本比本环境旧(matplotlib/scipy/urllib3 等)，本环境是能跑通的实测组合，没有强行降级。

## 4. 权重(已下好)

`simulation/weights/cg_weights/`:
- `groundingdino_swint_ogc.pth`(694MB)✅ 已下载
- `sam_vit_h_4b8939.pth`(2.4GB)✅ 已下载

FoundationPose 权重见 `weights/fp_weights/`(如需位姿估计)。

## 5. 环境变量

```bash
export GSA_PATH=<绝对路径>/simulation/src/perception/third_party/Grounded-Segment-Anything
```
`PhysicalValidator.__init__` 未设 `GSA_PATH` 会直接抛错。

## 6. 自检

```bash
conda activate brainary_sim
export GSA_PATH=.../Grounded-Segment-Anything
cd simulation
python -c "from engine_interface import PhysicalValidator; PhysicalValidator('config/global_config.yaml'); print('OK')"
```

## 6.1 全链路自检：`selftest_e2e.py`(不依赖 Isaac，纯合成 RGB-D)

```bash
conda activate brainary_sim && cd simulation
export GSA_PATH=$PWD/src/perception/third_party/Grounded-Segment-Anything
python selftest_e2e.py     # 预期 ~8s 后打印 success = True
```
实测：GDINO+SAM 从合成图检出 3 个刚体 -> 早融合建孪生世界 -> DAG 预演 grasp+place 全绿，
返回 `PASS` / `success=True`。把脚本里 `HALF` 调到 0.035(70mm 积木)则走另一条分支：
夹爪 80mm 极限下压坏积木 -> `REPLAN_REQUIRED` + `llm_reflection_prompt`(喂回规划 Agent 反思那条路)。

`verify_local_plan` 要多视角 RGB-D + 内外参，脚本用 SAPIEN 自己离屏渲染一张三色积木的图来喂它，
不需要 Isaac 那边在跑。几个必须对齐的约定：

- 深度单位是**毫米**(`cg_wrapper` 里 `depth_image[mask] / 1000.0`)；
- 外参是 **OpenCV 相机系** -> 世界的 4x4(x 右 y 下 z 前)。SAPIEN 的 `get_model_matrix()` 是 OpenGL 系，
  要右乘 `diag(1,-1,-1,1)` 转过去；
- 只有 `PHYSICS_DICTIONARY` 里的标签(red/blue/green cube、yellow box)会被留下，别的会被 `_clean_label` 丢掉。

## 6.2 ⚠️ 已修复：`scene.step()` 段错误

夹爪 `stiffness=2000` 硬挤 68mm 积木时，PhysX 默认的 `solver_velocity_iterations=1` 会解算发散，
`scene.step()` 直接段错误——**整个进程静默挂掉，没有 traceback**(只有 faulthandler 打出的 Python 栈)。
已在 `SceneBuilder.__init__` 里把速度迭代设成 4(建 Scene 前设置，见 `config/global_config.yaml`
的 `simulator_config`)。不动任何刚度参数，`predicted_force = squeeze_depth * k_gripper` 的业务模型不受影响。

> 排查笔记：这个崩溃对物体尺寸极度敏感(0.068m 不崩、0.06836m 必崩)，而且和感知/渲染**无关**——
> 把感知输出的几何存成 json 后单独回放(不加载 GDINO/SAM、不开 GPU 推理)照样崩。
> 另外**别**调高 `solver_position_iterations`：50 会让机械臂第一步就漂 2.3 rad 炸开，
> `(50, 1)` 甚至在只有一条空机械臂的场景里就直接段错误。

## 6.3 已知缺口：FoundationPose 分支不可用

`src/perception/wrappers/fp_wrapper.py` 里 `from utils.mesh import Mesh` 找不到模块——本仓库带的
FoundationPose 只有根目录的 `Utils.py`，没有 `utils/` 包(`from estimater import FoundationPose` 本身是好的)。
加上 `weights/fp_weights/` 里两个 checkpoint 也**没下载**，所以 `FoundationPoseWrapper` 目前是降级成
`estimator=None` 的直通模式。主管线没有任何地方用它，不影响 `verify_local_plan`。

## 7. 接入(见 INTEGRATION.md)

因与 Isaac 环境互斥,推荐把 `verify_adapter.verify_plan` 包成 HTTP 服务(类比 scene_describer:5599),
`run_brainary --verify` 通过网络调用;或在本 `brainary_sim` 环境内直接用 `verify_then_replan`。
