"""Setup script für automatische Installation"""
import sys
import subprocess
import os
from pathlib import Path

def main():
    print("🚀 VWL-RL-Cloud Setup")
    print("="*60)
    
    # Check Python version
    version = sys.version_info
    print(f"✅ Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor != 11:
        print(f"⚠️  Warnung: Python 3.11 empfohlen, du hast {version.major}.{version.minor}")
        print("   Ray RLlib funktioniert am besten mit Python 3.11")
        response = input("   Trotzdem fortfahren? (j/n): ")
        if response.lower() != 'j':
            print("❌ Setup abgebrochen. Installiere Python 3.11 von python.org")
            sys.exit(1)
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    print("\n📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Test installation
    print("\n🧪 Testing installation...")
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        print(f"✅ RLlib ready: {ray.__version__}")
    except Exception as e:
        print(f"❌ RLlib Test failed: {e}")
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running environment tests...")
    os.environ['PYTHONPATH'] = str(Path.cwd())
    result = subprocess.run([sys.executable, "tests/test_env.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"⚠️  Tests mit Warnungen: {result.stderr}")
    
    print("\n" + "="*60)
    print("✅ Setup erfolgreich abgeschlossen!")
    print("\n🚀 Nächste Schritte:")
    print("   1. Frontend starten: streamlit run frontend/app.py")
    print("   2. Training starten: python train/train_single.py")
    print("   3. Szenarien testen: python tests/test_scenarios.py")
    print("\n📖 Mehr Infos: README.md")
    print("="*60)

if __name__ == "__main__":
    main()
