#!/usr/bin/env python3

import os
import sys
import subprocess

def install_requirements():
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def check_env_vars():
    print("检查环境变量...")
    required_vars = ['ZHIPU_API_KEY', 'BASE_URL']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 缺少以下环境变量:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    else:
        print("✅ 环境变量配置正确！")
        return True

def main():
    print("GraphRAG 项目设置向导")
    print("=" * 40)
    
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    if not install_requirements():
        return False
    
    env_ok = check_env_vars()
    
    print("\n设置完成")
    print("=" * 40)
    
    if env_ok:
        print("🎉 可以运行: python demo.py")
    else:
        print("⚠️ 请先配置环境变量")
    
    return True

if __name__ == "__main__":
    main() 