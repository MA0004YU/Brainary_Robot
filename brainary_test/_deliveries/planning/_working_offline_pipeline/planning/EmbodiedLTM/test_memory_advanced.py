import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add current directory to sys.path so we can import memory_coordinator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_coordinator import initialize_rag, handle_insert, handle_query, handle_reset

FLAG_FILE = ".memory_initialized_flag"

async def test_advanced():
    print("=========================================")
    print("   Starting Memory Alignment Test")
    print("=========================================\n")
    
    # 1. Initialize Memory (Skip if already done)
    if not os.path.exists(FLAG_FILE):
        print("--- [1] Initializing and Inserting Memory ---")
        await handle_reset()
        
        print("Inserting Step 0: Agent sees a red CLEAN cup (1001)...")
        await handle_insert("在 Step 0 时，Agent 在厨房桌子上观察到了一个杯子（ID: 1001，颜色：红色，状态：CLEAN）。", None, None, None)
        
        print("Inserting Step 1: Agent pours coffee into cup 1001...")
        await handle_insert("在 Step 1 时，Agent 将咖啡倒入了杯子（ID: 1001）中。由于使用了该杯子，它的状态变为了 DIRTY。", None, None, None)
        
        print("Inserting Step 2: Cup 1001 disappears...")
        await handle_insert("在 Step 2 时，发生了一次环境刷新，放在桌子上的杯子（ID: 1001）突然消失不见了。", None, None, None)
        
        print("Inserting Step 3: Agent sees a new cup 2002...")
        await handle_insert("在 Step 3 时，Agent 打开橱柜，发现里面有一个新刷出来的杯子（ID: 2002，颜色：红色，状态：DIRTY）。", None, None, None)
        
        with open(FLAG_FILE, "w") as f:
            f.write("done")
        print("✅ Memory initialization completed.")
    else:
        print("--- [1] Skipping insertion, memory already initialized ---")
        await initialize_rag()
        print("✅ RAG Connections initialized.")

    # 2. Round 1: Check identity alignment
    print("\n--- [2] Query 1: Identity Alignment ---")
    query1 = "我在橱柜里看到一个新杯子（ID: 2002，颜色红色，状态DIRTY）。这个杯子是不是我之前倒过咖啡的那个杯子？"
    print(f"User Query: {query1}")
    res1 = await handle_query(query1, mode="hybrid", use_pm=False)
    ans1 = res1["final_answer"]
    print(f"Agent Response:\n{ans1}\n")
    
    try:
        assert "1001" in ans1, "Assertion Error: 未能找回旧ID 1001"
        assert "确认对齐" in ans1 or "同一" in ans1, "Assertion Error: 大模型未能得出是同一个物体的结论"
        print("✅ Query 1 Assertions Passed!")
    except AssertionError as e:
        print(f"❌ Query 1 Failed: {e}")
        
    # 3. Round 2: Update ID
    print("\n--- [3] Update Memory: Updating ID ---")
    print("Inserting update memory fact...")
    await handle_insert("系统确认记录：之前用来倒咖啡的那个原本放在桌子上的杯子（原ID: 1001），现在它的最新 ID 已经更新为 2002。", None, None, None)
    print("✅ Update memory inserted.")
    
    # 4. Round 3: Query the last used cup ID
    print("\n--- [4] Query 2: Final Used Cup Query ---")
    query3 = "请告诉我，我最后用来倒咖啡的那个杯子，它现在的最新 ID 是多少？"
    print(f"User Query: {query3}")
    res3 = await handle_query(query3, mode="hybrid", use_pm=False)
    ans3 = res3["final_answer"]
    print(f"Agent Response:\n{ans3}\n")
    
    try:
        assert "2002" in ans3, "Assertion Error: 模型未能输出更新后的最新 ID 2002"
        print("✅ Query 2 Assertions Passed!")
    except AssertionError as e:
        print(f"❌ Query 2 Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_advanced())
