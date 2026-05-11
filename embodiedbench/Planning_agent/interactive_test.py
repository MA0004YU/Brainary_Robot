import sys
import os
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.core_agent import IntentReasoningAgent
from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv

def run_interactive_test():
    parser = argparse.ArgumentParser(description="Intent Agent Interactive Test")
    parser.add_argument("--new_session", action="store_true", help="Start a new session, clearing short-term memory.")
    parser.add_argument("--instruction", type=str, help="Initial instruction to start a new session", default="")
    parser.add_argument("--resume", action="store_true", help="Resume the existing session.")
    parser.add_argument("--feedback", type=str, help="User feedback to resume an existing session", default="")
    parser.add_argument("--scene_id", type=str, help="Scene ID to load persistent knowledge", default="FloorPlan22")
    args = parser.parse_args()

    if not args.new_session and not args.resume:
        print("Please provide either --new_session or --resume.")
        return

    print("\n" + "="*50)
    print("   Starting Intent Reasoning Agent (Interactive)   ")
    print("="*50)

    # Initialize environment
    env = EBHabEnv(eval_set='base', selected_indexes=[3])
    
    if args.new_session:
        print(f"[*] Starting NEW session with instruction: {args.instruction}")
        instruction = args.instruction
        agent = IntentReasoningAgent(
            episode_id="test_interactive", 
            scene_id=args.scene_id
        )
        # Clear existing session memory if it's a new session
        agent.memory_manager.clear_session_context()
    else:
        print(f"[*] Resuming session with feedback: {args.feedback}")
        instruction = "RESUMED_SESSION" # Will be ignored by core_agent because global_intent is already loaded
        agent = IntentReasoningAgent(
            episode_id="test_interactive", 
            resume_feedback=args.feedback if args.feedback else "Continue",
            scene_id=args.scene_id
        )
        
    obs = env.reset()

    step_idx = 0
    done = False
    skill_set = env.language_skill_set
    observation_text = "You just spawned in the environment."
    
    while not done and step_idx < 50:
        # Save dual view (optional but good for logs)
        dual_img_path, fpv_img_path = agent.logger.save_obs_dual_view(obs, step_idx)
        print(f"\n---> [Step {step_idx}] Generated observation image at: {dual_img_path}")
        
        # Agent step
        validation_result = agent.step(
            instruction=instruction, 
            observation_text=observation_text, 
            skill_set=skill_set, 
            step_idx=step_idx, 
            img_path=fpv_img_path, 
            dual_img_path=dual_img_path
        )
        
        # Check if agent decided to stop and save session
        if validation_result.get("stop_and_save"):
            print(f"\n[!] Agent Execution Paused.")
            print(f"[!] Agent Message: {validation_result.get('communication_to_user')}")
            print("[!] Session state saved to session_context.md")
            print("[!] Exiting script. Run again with --resume --feedback '...' to resume.\n")
            break
            
        action_id = validation_result.get("action_id", 0)
        action_str = skill_set[action_id] if action_id < len(skill_set) else "UNKNOWN"
        print(f"---> Agent decided to execute: {action_str} (ID: {action_id})")
        
        # Execute action in environment
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
