import sys
import os
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.core_agent import IntentReasoningAgent
from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv

def run_interactive_test():
    parser = argparse.ArgumentParser(description="Intent Agent Interactive Test")
    parser.add_argument("--instruction", type=str, help="Task instruction", default="Get me a can of coke.")
    parser.add_argument("--scene_id", type=str, help="Scene ID for persistent knowledge", default="FloorPlan22")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("   Starting Intent Reasoning Agent (Interactive)   ")
    print("="*50)
    print(f"[*] Instruction: {args.instruction}")
    print(f"[*] Scene ID: {args.scene_id}")

    env = EBHabEnv(eval_set='base', selected_indexes=[3])
    agent = IntentReasoningAgent(
        episode_id="test_interactive",
        scene_id=args.scene_id,
    )

    obs = env.reset()
    instruction = args.instruction
    step_idx = 0
    done = False
    skill_set = env.language_skill_set
    observation_text = "You just spawned in the environment."

    while not done and step_idx < 50:
        dual_img_path, fpv_img_path = agent.logger.save_obs_dual_view(obs, step_idx)
        print(f"\n---> [Step {step_idx}] Generated observation image at: {dual_img_path}")

        validation_result = agent.step(
            instruction=instruction,
            observation_text=observation_text,
            skill_set=skill_set,
            step_idx=step_idx,
            img_path=fpv_img_path,
            dual_img_path=dual_img_path,
        )

        if validation_result.get("stop_and_save"):
            print(f"\n[!] Agent Execution Paused.")
            print(f"[!] Agent Message: {validation_result.get('communication_to_user')}")
            print("[!] Checkpoint saved to EmbodiedLTM. Exiting.\n")
            break

        action_id = validation_result.get("action_id", 0)
        action_str = skill_set[action_id] if action_id < len(skill_set) else "UNKNOWN"
        print(f"---> Agent decided to execute: {action_str} (ID: {action_id})")

        obs, reward, done, info = env.step(action_id)
        feedback = info.get('env_feedback', '')
        observation_text = f"Feedback: {feedback}."
        print(f"---> Environment Feedback: {feedback}")

        step_idx += 1
        time.sleep(1)

    print("\nScript Finished.")
    env.close()

if __name__ == "__main__":
    run_interactive_test()
