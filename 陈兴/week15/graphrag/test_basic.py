#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """测试基本模块导入"""
    print("测试模块导入...")
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import tiktoken
        from openai import OpenAI
        print("✅ 所有基础模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_graph_creation():
    """测试图创建功能"""
    print("测试图创建...")
    try:
        import networkx as nx
        
        # 创建测试图
        G = nx.Graph()
        G.add_node("李明", type="person", description="主角")
        G.add_node("张小雪", type="person", description="神秘女孩")
        G.add_edge("李明", "张小雪", relationship="合作关系")
        
        print(f"✅ 图创建成功，包含 {len(G.nodes())} 个节点和 {len(G.edges())} 条边")
        return True
    except Exception as e:
        print(f"❌ 图创建失败: {e}")
        return False

def test_text_processing():
    """测试文本处理功能"""
    print("测试文本处理...")
    try:
        import tiktoken
        
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # 测试文本
        test_text = "这是一个测试文本，用来验证文本处理功能是否正常工作。"
        tokens = encoding.encode(test_text)
        decoded = encoding.decode(tokens)
        
        print(f"✅ 文本处理成功，原文 {len(test_text)} 字符，编码为 {len(tokens)} 个token")
        return True
    except Exception as e:
        print(f"❌ 文本处理失败: {e}")
        return False

def test_file_reading():
    """测试文件读取"""
    print("测试文件读取...")
    try:
        with open("input/novel.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ 文件读取成功，内容长度: {len(content)} 字符")
        return True
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return False

def test_environment():
    """测试环境配置"""
    print("测试环境配置...")
    
    # 尝试加载环境变量
    load_dotenv()
    
    api_key = os.environ.get('ZHIPU_API_KEY')
    base_url = os.environ.get('BASE_URL')
    
    if api_key and base_url:
        print("✅ 环境变量配置完整")
        return True
    else:
        print("⚠️ 环境变量未完全配置（运行演示时需要配置）")
        return False

def test_graph_rag_class():
    """测试GraphRAG类的基本功能"""
    print("测试GraphRAG类...")
    
    try:
        # 临时设置环境变量（如果不存在）
        if not os.environ.get('ZHIPU_API_KEY'):
            os.environ['ZHIPU_API_KEY'] = 'test_key'
        if not os.environ.get('BASE_URL'):
            os.environ['BASE_URL'] = 'test_url'
        
        from graph_rag import GraphRAG
        
        # 创建实例（不会真正调用API）
        graph_rag = GraphRAG()
        
        # 测试文本分块功能
        test_text = "这是第一段。\n\n这是第二段。\n\n这是第三段。"
        chunks = graph_rag.chunk_text(test_text, chunk_size=50, overlap=10)
        
        print(f"✅ GraphRAG类创建成功，文本分块功能正常，生成 {len(chunks)} 个块")
        return True
    except Exception as e:
        print(f"❌ GraphRAG类测试失败: {e}")
        return False

def main():
    print("=" * 60)
    print("               GraphRAG 基础功能测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("图创建", test_graph_creation),
        ("文本处理", test_text_processing),
        ("文件读取", test_file_reading),
        ("环境配置", test_environment),
        ("GraphRAG类", test_graph_rag_class),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{passed + 1}/{total}] {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print("                    测试结果")
    print("=" * 60)
    print(f"总测试数: {total}")
    print(f"通过数: {passed}")
    print(f"失败数: {total - passed}")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
    elif passed >= total - 1:
        print("✅ 核心功能测试通过，系统基本可用。")
    else:
        print("⚠️ 部分测试失败，请检查环境配置。")
    
    print("\n下一步:")
    if passed >= total - 1:
        print("1. 配置环境变量（如果未配置）")
        print("2. 运行: python demo.py")
        print("3. 或运行: python graph_rag.py")
    else:
        print("1. 检查依赖安装: pip install -r requirements.txt")
        print("2. 重新运行测试: python test_basic.py")

if __name__ == "__main__":
    main() 