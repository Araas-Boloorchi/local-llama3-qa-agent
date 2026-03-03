import os
import subprocess
import sys
import time
import requests

def run_command(command, desc):
    print(f"\n[RUNNER] {desc}...")
    try:
        subprocess.check_call(command, shell=True)
        print(f"[RUNNER] {desc} completed successfully.")
        return True
    except subprocess.CalledProcessError:
        print(f"[RUNNER] {desc} FAILED.")
        return False

def check_server(url="http://localhost:3000"):
    print(f"[RUNNER] Waiting for server at {url}...")
    for _ in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("[RUNNER] Server is UP and responding!")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("[RUNNER] Server failed to respond.")
    return False

def main():
    print("=== QA Agent Local Runner ===")
    
    # 1. Setup / Download Model
    if ignored := not os.path.exists(os.path.join("models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")):
        if not run_command("python setup_local.py", "Downloading Model"):
            return
    else:
        print("[RUNNER] Model already exists. Skipping download.")

    # 2. Test Agent
    print("\n[RUNNER] Testing Agent Logic...")
    if not run_command("python agent.py", "Running Agent Tests"):
        print("Agent self-test failed. Please check logs.")
        # We might continue or stop, but let's stop to be safe
        # return 

    # 3. Start Server
    print("\n[RUNNER] Starting API Server...")
    # we use Popen to run in background or separate process
    server_process = subprocess.Popen(["python", "server.py"], shell=False)
    
    # 4. Check Health
    if check_server():
        print("\n[RUNNER] Application is running!")
        print("Backend: http://localhost:3000")
        print("Frontend: http://localhost:3000")
        print("Press Ctrl+C to stop the server.")
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n[RUNNER] Stopping server...")
            server_process.terminate()
    else:
        print("\n[RUNNER] Server failed to start.")
        server_process.terminate()

if __name__ == "__main__":
    main()
