#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graph_rag import GraphRAG
import os

def main():
    print("=" * 60)
    print("           小说《时光之城》GraphRAG 知识图谱演示")
    print("=" * 60)
    
    # 检查环境变量
    if not os.environ.get('ZHIPU_API_KEY'):
        print("错误：请先设置 ZHIPU_API_KEY 环境变量")
        return
    
    # 初始化GraphRAG
    print("\n1. 初始化GraphRAG系统...")
    graph_rag = GraphRAG()
    
    # 显示当前工作目录和文件检查
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目目录: {graph_rag.project_dir}")
    
    # 检查输入文件是否存在
    input_file = "input/novel.txt"
    if os.path.exists(input_file):
        print(f"✓ 找到输入文件: {input_file}")
    else:
        print(f"✗ 输入文件不存在: {input_file}")
        if os.path.exists("input"):
            print(f"input目录内容: {os.listdir('input')}")
        else:
            print("input目录不存在")
        return
    
    # 构建知识图谱
    print("\n2. 构建知识图谱...")
    graph_rag.build_graph(input_file)
    
    # 保存结果
    print("\n3. 保存结果...")
    graph_rag.save_results()
    
    # 可视化图谱
    print("\n4. 生成图谱可视化...")
    graph_rag.visualize_graph("output/knowledge_graph.png")
    
    # 演示查询
    print("\n5. 演示查询功能...")
    
    # 本地搜索示例
    print("\n--- 本地搜索示例 ---")
    local_queries = [
        "李明是谁？",
        "时间管理局的作用是什么？",
        "张小雪的特点？"
    ]
    
    for query in local_queries:
        print(f"\n问题: {query}")
        answer = graph_rag.local_search(query)
        print(f"答案: {answer}")
        print("-" * 40)
    
    # 全局搜索示例
    print("\n--- 全局搜索示例 ---")
    global_queries = [
        "整个故事的主要冲突是什么？",
        "小说中有哪些重要的组织机构？",
        "时空穿越在故事中起到什么作用？"
    ]
    
    for query in global_queries:
        print(f"\n问题: {query}")
        answer = graph_rag.global_search(query)
        print(f"答案: {answer}")
        print("-" * 40)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("                    统计信息")
    print("=" * 60)
    print(f"总实体数: {len(graph_rag.entities)}")
    print(f"总关系数: {len(graph_rag.relationships)}")
    print(f"社区数: {len(graph_rag.communities)}")
    print(f"文本块数: {len(graph_rag.chunks)}")
    
    # 显示实体类型分布
    entity_types = {}
    for entity in graph_rag.entities.values():
        entity_type = entity.get("type", "unknown")
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print("\n实体类型分布:")
    for entity_type, count in entity_types.items():
        print(f"  {entity_type}: {count}")
    
    print("\n图谱构建完成！")
    print("结果文件保存在 output/ 目录中")
    print("- entities.csv: 实体信息")
    print("- relationships.csv: 关系信息") 
    print("- communities.json: 社区信息")
    print("- knowledge_graph.png: 图谱可视化")

if __name__ == "__main__":
    main() 