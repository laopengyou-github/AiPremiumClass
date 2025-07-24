#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graph_rag import GraphRAG
import os

def test_visualization():
    """测试可视化功能"""
    print("测试可视化功能...")
    
    try:
        # 初始化GraphRAG
        print("1. 初始化GraphRAG...")
        graph_rag = GraphRAG()
        
        # 构建简化的图谱（避免重复构建）
        if os.path.exists("output/entities.csv"):
            print("2. 发现已有图谱数据，跳过构建...")
            # 这里可以加载已有数据，但为了简化测试，我们重新构建
        
        print("2. 构建知识图谱...")
        graph_rag.build_graph("input/novel.txt")
        
        # 测试可视化
        print("3. 生成可视化图谱...")
        graph_rag.visualize_graph("output/test_graph.png")
        
        print("✓ 可视化测试完成！")
        print("  图片已保存到 output/test_graph.png")
        
    except Exception as e:
        print(f"✗ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization() 