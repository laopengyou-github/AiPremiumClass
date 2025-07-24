#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graph_rag import GraphRAG
import os

def test_search():
    """测试搜索功能"""
    print("测试搜索功能修复...")
    
    # 检查环境变量
    if not os.environ.get('ZHIPU_API_KEY'):
        print("错误：请先设置 ZHIPU_API_KEY 环境变量")
        return
    
    try:
        # 初始化GraphRAG
        print("1. 初始化GraphRAG...")
        graph_rag = GraphRAG()
        
        # 构建图谱
        print("2. 构建知识图谱...")
        graph_rag.build_graph("input/novel.txt")
        
        # 显示实体信息
        print(f"\n图谱统计:")
        print(f"- 实体数: {len(graph_rag.entities)}")
        print(f"- 关系数: {len(graph_rag.relationships)}")
        print(f"- 社区数: {len(graph_rag.communities)}")
        
        # 显示前5个实体
        print(f"\n前5个实体:")
        for i, (name, data) in enumerate(graph_rag.entities.items()):
            if i >= 5:
                break
            print(f"- {name}: {data.get('type', 'unknown')} - {data.get('description', '')[:50]}...")
        
        # 测试本地搜索（仅搜索，不调用API）
        print(f"\n3. 测试本地搜索关键词匹配...")
        
        # 简单的实体查找测试
        queries = ["李明", "张小雪", "时间", "管理"]
        
        for query in queries:
            print(f"\n查询: {query}")
            # 手动测试关键词匹配逻辑
            import re
            keywords = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', query)
            print(f"提取的关键词: {keywords}")
            
            found_entities = []
            for entity_name, entity_data in graph_rag.entities.items():
                entity_name_lower = entity_name.lower()
                entity_desc = entity_data.get("description", "").lower()
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if (keyword_lower in entity_name_lower or 
                        keyword_lower in entity_desc or
                        entity_name_lower in keyword_lower):
                        found_entities.append(entity_name)
                        break
            
            print(f"匹配的实体: {found_entities}")
        
        print(f"\n✓ 搜索功能测试完成！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search() 