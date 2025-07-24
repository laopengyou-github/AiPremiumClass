import os
import json
import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Tuple, Any
import tiktoken

class GraphRAG:
    def __init__(self, config_path="settings.yaml"):
        load_dotenv()
        
        # 设置正确的工作目录
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.project_dir)
        
        self.client = OpenAI(
            api_key=os.environ.get('ZHIPU_API_KEY'),
            base_url=os.environ.get('BASE_URL')
        )
        self.model = "glm-4-flash-250414"
        self.embedding_model = "embedding-3"
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # 图数据结构
        self.graph = nx.Graph()
        self.entities = {}
        self.relationships = []
        self.communities = {}
        self.chunks = []
        
    def load_text(self, file_path: str) -> str:
        """加载文本文件"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            # 尝试在项目目录中查找
            full_path = os.path.join(self.project_dir, file_path)
            if os.path.exists(full_path):
                file_path = full_path
            else:
                raise FileNotFoundError(f"找不到文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 100) -> List[str]:
        """将文本分块"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks
    
    def load_prompt(self, prompt_name: str) -> str:
        """加载prompt模板"""
        prompt_path = f"prompts/{prompt_name}.txt"
        
        # 检查文件是否存在
        if not os.path.exists(prompt_path):
            # 尝试在项目目录中查找
            full_path = os.path.join(self.project_dir, prompt_path)
            if os.path.exists(full_path):
                prompt_path = full_path
            else:
                raise FileNotFoundError(f"找不到prompt文件: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_entities_and_relationships(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """从文本中提取实体和关系"""
        prompt_template = self.load_prompt("entity_extraction")
        prompt = prompt_template.replace("{input_text}", text).replace("{tuple_delimiter}", "|")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            content = response.choices[0].message.content
            
            # 解析实体
            entities = []
            entity_pattern = r'\("entity"\|([^|]+)\|([^|]+)\|([^)]+)\)'
            entity_matches = re.findall(entity_pattern, content)
            
            for match in entity_matches:
                entities.append({
                    "name": match[0].strip(),
                    "type": match[1].strip(),
                    "description": match[2].strip()
                })
            
            # 解析关系
            relationships = []
            rel_pattern = r'\("relationship"\|([^|]+)\|([^|]+)\|([^|]+)\|([^)]+)\)'
            rel_matches = re.findall(rel_pattern, content)
            
            for match in rel_matches:
                relationships.append({
                    "source": match[0].strip(),
                    "target": match[1].strip(),
                    "description": match[2].strip(),
                    "strength": float(match[3].strip()) if match[3].strip().isdigit() else 5.0
                })
            
            return entities, relationships
            
        except Exception as e:
            print(f"提取实体和关系时出错: {e}")
            return [], []
    
    def build_graph(self, input_file: str):
        """构建知识图谱"""
        print("正在加载文本...")
        text = self.load_text(input_file)
        
        print("正在分块文本...")
        self.chunks = self.chunk_text(text)
        
        print(f"共分为 {len(self.chunks)} 个块，正在提取实体和关系...")
        
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(self.chunks):
            print(f"处理第 {i+1}/{len(self.chunks)} 个块...")
            entities, relationships = self.extract_entities_and_relationships(chunk)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # 合并重复的实体
        entity_dict = {}
        for entity in all_entities:
            name = entity["name"]
            if name in entity_dict:
                # 合并描述
                entity_dict[name]["description"] += f"; {entity['description']}"
            else:
                entity_dict[name] = entity
        
        self.entities = entity_dict
        self.relationships = all_relationships
        
        # 构建NetworkX图
        for entity_name, entity_data in self.entities.items():
            self.graph.add_node(entity_name, **entity_data)
        
        for rel in self.relationships:
            if rel["source"] in self.entities and rel["target"] in self.entities:
                self.graph.add_edge(
                    rel["source"], 
                    rel["target"], 
                    description=rel["description"],
                    weight=rel["strength"]
                )
        
        print(f"图谱构建完成！包含 {len(self.entities)} 个实体和 {len(self.relationships)} 个关系。")
        
        # 发现社区
        self.detect_communities()
    
    def detect_communities(self):
        """检测社区"""
        try:
            communities = nx.community.greedy_modularity_communities(self.graph)
            
            for i, community in enumerate(communities):
                community_entities = list(community)
                self.communities[f"community_{i}"] = {
                    "entities": community_entities,
                    "size": len(community_entities)
                }
            
            print(f"发现 {len(communities)} 个社区")
        except Exception as e:
            print(f"社区检测失败: {e}")
    
    def generate_community_summary(self, community_id: str) -> str:
        """生成社区摘要"""
        if community_id not in self.communities:
            return "社区不存在"
        
        community = self.communities[community_id]
        entities = community["entities"]
        
        # 获取社区内的关系
        community_relationships = []
        for rel in self.relationships:
            if rel["source"] in entities and rel["target"] in entities:
                community_relationships.append(rel)
        
        # 构建社区报告
        entity_list = []
        for entity_name in entities:
            if entity_name in self.entities:
                entity_info = self.entities[entity_name]
                entity_list.append(f"{entity_name} ({entity_info['type']}): {entity_info['description']}")
        
        relationship_list = []
        for rel in community_relationships:
            relationship_list.append(f"{rel['source']} -> {rel['target']}: {rel['description']}")
        
        prompt_template = self.load_prompt("community_report")
        prompt = prompt_template.replace("{entity_list}", "\n".join(entity_list))
        prompt = prompt_template.replace("{relationship_list}", "\n".join(relationship_list))
        prompt = prompt.replace("{max_length}", "2000")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成社区摘要时出错: {e}"
    
    def local_search(self, query: str, top_k: int = 5) -> str:
        """本地搜索：基于实体匹配"""
        # 寻找相关实体
        relevant_entities = []
        query_lower = query.lower()
        
        # 提取查询中的关键词（简单分词）
        import re
        keywords = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', query)  # 提取中文和英文词
        
        for entity_name, entity_data in self.entities.items():
            entity_name_lower = entity_name.lower()
            entity_desc = entity_data.get("description", "").lower()
            
            # 检查是否有关键词匹配
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if (keyword_lower in entity_name_lower or 
                    keyword_lower in entity_desc or
                    entity_name_lower in keyword_lower):
                    relevant_entities.append((entity_name, entity_data))
                    break
        
        if not relevant_entities:
            return "没有找到相关实体"
        
        # 获取相关关系
        relevant_relationships = []
        entity_names = [e[0] for e in relevant_entities]
        
        for rel in self.relationships:
            if rel["source"] in entity_names or rel["target"] in entity_names:
                relevant_relationships.append(rel)
        
        # 构建答案
        context = "相关实体:\n"
        for entity_name, entity_data in relevant_entities[:top_k]:
            context += f"- {entity_name} ({entity_data['type']}): {entity_data['description']}\n"
        
        context += "\n相关关系:\n"
        for rel in relevant_relationships[:top_k]:
            context += f"- {rel['source']} -> {rel['target']}: {rel['description']}\n"
        
        # 使用LLM生成回答
        prompt = f"""基于以下知识图谱信息回答问题：

问题: {query}

相关信息:
{context}

请基于上述信息提供详细的中文回答："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {e}"
    
    def global_search(self, query: str) -> str:
        """全局搜索：基于社区摘要"""
        if not self.communities:
            return "没有发现社区，无法进行全局搜索"
        
        # 生成所有社区的摘要
        community_summaries = []
        for community_id in self.communities:
            summary = self.generate_community_summary(community_id)
            community_summaries.append(f"社区 {community_id}:\n{summary}")
        
        # 使用社区摘要回答问题
        context = "\n\n".join(community_summaries)
        
        prompt = f"""基于以下知识图谱的社区摘要信息回答问题：

问题: {query}

社区摘要信息:
{context}

请基于上述信息提供全面的中文回答，综合考虑不同社区的信息："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {e}"
    
    def visualize_graph(self, save_path: str = "output/knowledge_graph.png"):
        """可视化图谱"""
        # 确保输出目录存在
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.figure(figsize=(15, 10))
        
        # 设置布局
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # 按类型设置节点颜色
        node_colors = []
        for node in self.graph.nodes():
            entity_type = self.entities[node].get("type", "unknown")
            if entity_type == "person":
                node_colors.append("lightblue")
            elif entity_type == "organization":
                node_colors.append("lightgreen")
            elif entity_type == "location":
                node_colors.append("orange")
            elif entity_type == "event":
                node_colors.append("pink")
            elif entity_type == "time":
                node_colors.append("yellow")
            else:
                node_colors.append("gray")
        
        # 绘制图
        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=1000,
                font_size=8,
                font_weight="bold",
                with_labels=True,
                edge_color="gray",
                alpha=0.7)
        
        plt.title("小说知识图谱", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"图谱已保存到 {save_path}")
    
    def save_results(self):
        """保存结果"""
        # 确保输出目录存在
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存实体
        entities_df = pd.DataFrame.from_dict(self.entities, orient='index')
        entities_df.to_csv("output/entities.csv", encoding='utf-8')
        
        # 保存关系
        relationships_df = pd.DataFrame(self.relationships)
        relationships_df.to_csv("output/relationships.csv", encoding='utf-8')
        
        # 保存社区信息
        with open("output/communities.json", 'w', encoding='utf-8') as f:
            json.dump(self.communities, f, ensure_ascii=False, indent=2)
        
        print("结果已保存到 output/ 目录")

if __name__ == "__main__":
    # 初始化GraphRAG
    graph_rag = GraphRAG()
    
    # 构建图谱
    graph_rag.build_graph("input/novel.txt")
    
    # 可视化图谱
    graph_rag.visualize_graph("output/knowledge_graph.png")
    
    # 保存结果
    graph_rag.save_results()
    
    print("\n" + "="*50)
    print("GraphRAG 构建完成！")
    print("="*50)
    
    # 交互式问答
    while True:
        print("\n请选择搜索类型：")
        print("1. 本地搜索 (基于实体)")
        print("2. 全局搜索 (基于社区)")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == "3":
            break
        elif choice in ["1", "2"]:
            query = input("\n请输入问题: ").strip()
            if not query:
                continue
                
            print("\n正在生成回答...")
            
            if choice == "1":
                answer = graph_rag.local_search(query)
                print(f"\n【本地搜索结果】\n{answer}")
            else:
                answer = graph_rag.global_search(query)
                print(f"\n【全局搜索结果】\n{answer}")
        else:
            print("无效选择，请重试") 