# lianyu — 感知模块(Perception)

与 `projects/xiaoyu/`(记忆模块 Brainary memory)平级的**感知模块**汇总。把场景图像变成结构化的
物体识别 / 场景理解结果,供记忆模块和规划模块使用。

> 这是**汇总副本**(非破坏性):接进 Isaac 仿真 UI 的同一套代码仍在
> `projects/franka_v1_skill_lab/{perception_qwen,scene_describer}/`(被 `test_mode_ui.py --describe` 引用)。
> lianyu 用于把感知模块作为独立交付件集中存放,并对接 xiaoyu 记忆模块。

## 目录

```
lianyu/
├── perception_vlm.py        ← 核心:别人开发的 Qwen2.5-VL-3B 本地识别器(与 xiaoyu 内那份一致)
│                              recognize_batch(image_paths, candidate_labels) -> List[RecognitionResult]
│                              (辅助视图 + 低置信复核 + JSON 外部记忆的推理时 harness)
├── perception_qwen/         ← 后端A:把 perception_vlm 包成 HTTP server(:5601 /recognize)+ isaac 端客户端
│   ├── qwen_perception_server.py / qwen_client.py / headless_test.py / README.md ...
└── scene_describer/         ← 后端B:GPT-5.5(中转 Responses API)双图+状态 -> {objects, relations}(:5599 /describe)
    ├── vlm_describe_server.py / describer_client.py / describe_panel.py / schema.py ...
```

两个后端对应两条路线:**本地小模型(Qwen,占 GPU)** vs **云端大模型(GPT,走 API)**。
仿真 UI 面板 "Scene Perception (GPT / Qwen)" 可下拉切换。

## 输出契约

- **Qwen**(`RecognitionResult`,每图一个):`primary_label / confidence / objects[] / attributes{color,shape,texture} / scene / reasoning / uncertainty`。
- **GPT**(每次一个):`scene_summary / objects[] / relations[]`(JSON Schema 强约束)。

---

## 与记忆模块(xiaoyu)的连接 —— 结论:本来就是为对接而设计的 ✅

`xiaoyu/brainary_memory_pkg` 是「感知 → 记忆 → 规划」桥,**自带一份相同的 `perception_vlm.py`**,
并提供现成胶水:

```
RGB 图像
  └─ QwenVLMPerception.recognize_batch()           ← perception_vlm.py(= 本模块核心)
       └─ VisionPerceptionAdapter.perceive()       ← memory_module/perception_adapter.py
            ├─ encode_recognition_results() -> (T,P,D) 特征  -> memory.on_perception_features()
            └─ update_observation(visible_objects, scene_text, attributes)
                 └─ 三层记忆(working / episodic / semantic)  ← embodiedbench/memory_manip/
                      └─ PlanningContext                      ← memory_module/planning_interface.py
                           └─ 规划模块 / VLM Brain
```

**单一入口** `PerceptionMemoryPipeline`(`memory_module/pipeline.py`):

```python
from memory_module import PerceptionMemoryPipeline
pipe = PerceptionMemoryPipeline.create(store_dir="memory_store/")   # 内部自建 QwenVLMPerception
pipe.session_start(); pipe.begin_episode("ep001", "pick up the cube")
pipe.process_perception(["rgb.png"], candidate_labels=["cube","target"], scene_state={...})  # 感知->记忆
ctx = pipe.get_planning_context()      # 记忆汇总给规划:visible_objects/recommended_skills/similar_episodes...
pipe.record_action("grasp", success=True)
pipe.end_episode(success=True); pipe.session_end()
```

适配器**直接吃 `RecognitionResult`**(perception_vlm 的原生输出),所以本模块的 `perception_vlm.py`
零改动即可接上记忆。

### 接到我们的 Isaac 仿真:三种接法(由浅入深)

1. **进程内直连(最简单)**:在已有 Isaac 进程里,把抓到的 front/wrist 帧存临时 PNG ->
   `pipe.process_perception([png])`。复用 `perception_qwen` 的 `_cap_rgb` 抓帧 + `headless_test.py` 的场景启动。
   注意:`PerceptionMemoryPipeline.create()` 会**再加载一份 Qwen 到本进程**(和仿真抢显存)。

2. **复用已起的 Qwen server(省显存,推荐)**:记忆模块要的是 `RecognitionResult`,而我们的
   `:5601 /recognize` 已经返回等价 dict。可写一个 thin perceiver 适配器:把 `recognize()` 的 dict
   反序列化回 `RecognitionResult`,用 `PerceptionMemoryPipeline.create_from_existing(perceiver, memory)`
   注入 —— 这样 Qwen 只加载一次(在 :5601),记忆进程不占额外显存。

3. **文件协议(松耦合)**:按 `INTEGRATION_GUIDE.md` 的 `scene_state.json` / `memory_context.json` /
   `result.json` 三文件协议,Isaac 与记忆 pipeline 分进程跑。适合接入规划/VLM Brain 的完整闭环。

### scene_state 对接点
记忆 pipeline 的 `process_perception(scene_state=...)` 吃的字段(cube_pose/target_pose/ee_pose/
gripper_width/robot_joint_pos/step_index)我们这边都能从 `SceneState` / `collect_state()` 取到,做一层
字段映射即可把仿真 GT 喂进 working memory。

> 详细 API 见 `xiaoyu/brainary_memory_pkg/memory_module/INTEGRATION_GUIDE.md`。
