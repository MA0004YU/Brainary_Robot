import os
import logging
from datetime import datetime

class AgentLogger:
    def __init__(self, episode_id=None, log_dir=None):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_id = episode_id if episode_id else run_timestamp
        
        if log_dir and os.path.exists(log_dir):
            self.run_dir = log_dir
        else:
            base_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            self.run_dir = os.path.join(base_log_dir, f"run_{self.episode_id}_{run_timestamp}")
            
        self.visuals_dir = os.path.join(self.run_dir, "visuals")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.visuals_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.run_dir, "agent_decision.log")
        self.run_summary_file = os.path.join(self.run_dir, "run_summary.md")
        
        # Initialize the markdown summary only if it's a new log dir
        if not (log_dir and os.path.exists(self.run_summary_file)):
            with open(self.run_summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# Intent Reasoning Agent Run Summary\n")
                f.write(f"**Episode ID:** {self.episode_id}  \n")
                f.write(f"**Timestamp:** {run_timestamp}  \n\n---\n\n")
        
        # Configure logging
        self.logger = logging.getLogger(f"IntentAgent_{self.episode_id}_{run_timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Avoid duplicate logs if instantiated multiple times
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def save_step_image(self, original_img_path: str, step: int) -> str:
        """将环境原始截图拷贝到本次运行专属的视觉目录，并返回新路径"""
        if not original_img_path or not os.path.exists(original_img_path):
            return None
        import shutil
        new_filename = f"step_{step:02d}_obs.png"
        new_path = os.path.join(self.visuals_dir, new_filename)
        shutil.copy(original_img_path, new_path)
        return new_path

    def save_obs_dual_view(self, obs, step: int) -> tuple:
        """提取 head_rgb 和 third_rgb，拼接成一张包含双视角的图片"""
        from embodiedbench.envs.eb_habitat.utils import observations_to_image
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        img_head = Image.fromarray(observations_to_image(obs, info={}, key="head_rgb"))
        
        # 尝试提取 third_rgb
        third_key = None
        for k in obs.keys():
            if "third_rgb" in k and hasattr(obs[k], 'shape') and len(obs[k].shape) > 1:
                third_key = k
                break

        if third_key:
            img_third = Image.fromarray(observations_to_image(obs, info={}, key=third_key))
            # Resize if height mismatch
            if img_head.height != img_third.height:
                img_third = img_third.resize((int(img_third.width * img_head.height / img_third.height), img_head.height))
            
            # Stitch horizontally
            total_width = img_head.width + img_third.width
            max_height = max(img_head.height, img_third.height)
            merged_img = Image.new('RGB', (total_width, max_height))
            merged_img.paste(img_head, (0, 0))
            merged_img.paste(img_third, (img_head.width, 0))
            
            # Optional: Add text labels
            try:
                draw = ImageDraw.Draw(merged_img)
                draw.text((10, 10), "First Person (head_rgb)", fill="red")
                draw.text((img_head.width + 10, 10), "Third Person (third_rgb)", fill="red")
            except Exception:
                pass
                
            final_img = merged_img
        else:
            final_img = img_head

        new_filename = f"step_{step:02d}_obs_dual.png"
        new_path = os.path.join(self.visuals_dir, new_filename)
        final_img.save(new_path)
        
        fpv_filename = f"step_{step:02d}_obs_fpv.png"
        fpv_path = os.path.join(self.visuals_dir, fpv_filename)
        img_head.save(fpv_path)
        
        return new_path, fpv_path

    def log_ground_truth(self, gt_info: dict):
        """将当前关卡的 Ground Truth (物品所在位置) 写入日志开头"""
        import json
        with open(self.run_summary_file, 'a', encoding='utf-8') as f:
            f.write("### Ground Truth Locations\n\n")
            f.write("```json\n")
            f.write(json.dumps(gt_info, indent=2, ensure_ascii=False) + "\n")
            f.write("```\n\n---\n\n")

    def log_module_output(self, module_name: str, step: int, output_data: str, img_path: str = None):
        """记录特定模块在特定步骤的核心输出，(已移除了冗余的 Markdown JSON 写入)"""
        separator = "-" * 40
        log_message = f"\n{separator}\n[Step: {step}] [Module: {module_name}]\n{output_data}\n{separator}"
        self.logger.info(log_message)

    def log_step_markdown(self, step: int, img_path: str, held_object: str, legal_combos_count: int, visited_locations: dict, memory_objects: dict, ranked_plans: list, selected_action: str):
        """写入极简版的 Step Markdown 战报"""
        with open(self.run_summary_file, 'a', encoding='utf-8') as f:
            f.write(f"### Step {step}\n\n")
            if img_path and os.path.exists(img_path):
                rel_img_path = os.path.relpath(img_path, self.run_dir)
                f.write(f"![Observation]({rel_img_path})\n\n")
            
            f.write(f"- **手持物品 (Held Object)**: `{held_object}`\n")
            f.write(f"- **可选动作数量**: `{legal_combos_count}` 个\n")
            f.write(f"- **已探索地点**: `{visited_locations}`\n")
            f.write(f"- **记忆物品**: `{list(memory_objects.keys())}`\n")
            f.write("- **Ranker 决策 (Top-2)**:\n")
            for i, plan in enumerate(ranked_plans[:2]):
                f.write(f"  - Plan {i+1}: {plan.get('plan_description')} (Score: {plan.get('rank_score')})\n")
            f.write(f"- **最终执行**: `{selected_action}`\n\n---\n\n")

    def info(self, msg: str):
        self.logger.info(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
