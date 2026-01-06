# ODRL_Check.py

import os
import json
import pyshacl
from rdflib import Graph, Namespace, RDF

from rdflib import Namespace, RDF
SH = Namespace("http://www.w3.org/ns/shacl#")

DEFAULT_SHACL_FILE_PATH = r"ODRL_V3\data preparation\shacl_for_validation\ODRL_Rule_Shapes.ttl"

def count_shacl_constraints(shacl_graph):
    """
    统计SHACL图中定义的顶级约束块数量。
    一个顶级约束块是指直接附加到sh:NodeShape或sh:PropertyShape上的sh:property、
    sh:or、sh:and、sh:xone、sh:not等约束。
    逻辑组合（如sh:or）本身被视为一个约束，其内部的子约束不单独计数。
    """
    constraint_blocks = set()
    
    # 1. 找出所有的NodeShape和PropertyShape。
    # 这些是SHACL图中的主要形状定义。
    all_shapes = set(shacl_graph.subjects(RDF.type, SH.NodeShape))
    all_shapes.update(shacl_graph.subjects(RDF.type, SH.PropertyShape))

    # 2. 遍历每个主要形状，收集其直接定义的顶级约束。
    for s in all_shapes:
        # 收集直接附加的 sh:property 约束块
        for o in shacl_graph.objects(s, SH.property):
            constraint_blocks.add(o)
        
        # 收集直接附加的逻辑组合约束块 (sh:or, sh:and, sh:xone, sh:not)
        # 这些逻辑组合本身被视为一个顶级约束
        for o in shacl_graph.objects(s, SH["or"]):
            constraint_blocks.add(o)
        for o in shacl_graph.objects(s, SH["and"]):
            constraint_blocks.add(o)
        for o in shacl_graph.objects(s, SH.xone):
            constraint_blocks.add(o)
        for o in shacl_graph.objects(s, SH["not"]):
            constraint_blocks.add(o)

    return len(constraint_blocks)

def validate_odrl_against_shacl(odrl_content_str:str, shacl_ttl_path: str = DEFAULT_SHACL_FILE_PATH):
    """
    Validates the given ODRL Turtle file against the SHACL shapes.
    
    :return: Tuple (conforms, total_constraints, num_violations, violation_reports)
    """
    SH = Namespace("http://www.w3.org/ns/shacl#")
    
    # Load the ODRL data
    odrl_graph = Graph()
    try:
        # 从字符串加载ODRL数据
        odrl_graph.parse(data=odrl_content_str, format='json-ld')
    except Exception as e:
        error_msg = f"ODRL内容解析错误: {e}"
        print(error_msg)
        error_report = {
            'result_path': 'N/A (Parsing Error)',
            'message': f"Failed to parse ODRL content as JSON-LD. Error: {e}",
            'constraint_component': 'N/A (Parsing Error)',
        }
        # --- [修改] ---
        # 返回 -1 表示解析阶段就已失败，无法进行验证
        return False, 0, -1, [error_report]

    # Load the SHACL shapes
    shacl_graph = Graph()
    try:
        shacl_graph.parse(shacl_ttl_path, format='ttl')
        print("SHACL graph parsed successfully.")
    except Exception as e:
        print(f"Error parsing SHACL shapes: {e}")
        return None, None, None, None
    
    total_constraints = count_shacl_constraints(shacl_graph)
    
    try:
        # 执行SHACL验证
        conforms, results_graph, _ = pyshacl.validate(
            data_graph=odrl_graph,
            shacl_graph=shacl_graph,
            inference='rdfs',
            abort_on_first=False,
            meta_shacl=False,
            advanced=True,
            debug=False
        )
    except Exception as e:
        print(f"CRITICAL: pyshacl validation crashed due to malformed data: {e}")
        error_report = {
            'result_path': 'N/A (Structural Error)',
            'message': f"Validation failed due to a fundamental data structure error. The ODRL data is likely not valid RDF. Error: {e}",
            'constraint_component': 'N/A (Structural Error)',
        }
        # --- [修改] ---
        # 返回 -1 表示验证过程中崩溃，无法统计违规数量
        return False, total_constraints, -1, [error_report]
    
    # (后续代码保持不变)
    violated_paths = set()
    violation_reports = []
    
    validation_results = list(results_graph.subjects(RDF.type, SH.ValidationResult))
    
    for result in validation_results:
        severity = results_graph.value(result, SH.resultSeverity)
        if severity != SH.Violation:
            continue

        path = results_graph.value(result, SH.resultPath)
        path_str = str(path) if path else "N/A"
        
        violated_paths.add(path_str)
        
        report = {
            'result_path': path_str,
            'message': str(results_graph.value(result, SH.resultMessage)),
            'constraint_component': str(results_graph.value(result, SH.sourceConstraintComponent)),
        }
        violation_reports.append(report)
    
    num_violations = len(violated_paths)
    
    return conforms, total_constraints, num_violations, violation_reports

if __name__ == '__main__':
    # 请确保 'test' 文件夹存在，并且 'odrl.ttl' 和 'shacl.ttl' 文件放在里面
    odrl_ttl_data = """
# Based on the provided guidelines and details, here is a comprehensive ODRL policy for the DE_Staatstheater_Augsburg's collaboration with a local university for an educational program on theater history and cultural heritage. The policy grants access to the 'HistoricalArchives' asset for free scientific research:

# ```ttl
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix dc: <http://purl.org/dc/terms/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix drk: <http://w3id.org/drk/> .
@prefix vcard: <http://www.w3.org/2006/vcard/ns#> .

# Define the DE_Staatstheater_Augsburg as a Party
drk:DE_Staatstheater_Augsburg a odrl:Party, vcard:Organization ;
    odrl:uid <http://w3id.org/drk/DE_Staatstheater_Augsburg> ;
    dc:description "State Theater Augsburg, Germany"^^xsd:string .

# Define the local university as a Party
drk:LocalUniversity a odrl:Party, vcard:Organization ;
    odrl:uid <http://w3id.org/drk/LocalUniversity> ;
    dc:description "Local University collaborating for educational purposes"^^xsd:string .

# Define the asset 'HistoricalArchives'
drk:HistoricalArchives a odrl:Asset ;
    odrl:uid <http://w3id.org/drk/HistoricalArchives> ;
    dc:description "Digitized historical data on theater history and cultural heritage"^^xsd:string .

# Define the Agreement
drk:EducationalProgramAgreement a odrl:Agreement, odrl:Policy ;
    odrl:uid <http://w3id.org/drk/EducationalProgramAgreement> ;
    dc:creator "DE_Staatstheater_Augsburg"^^xsd:string ;
    dc:description "Agreement for providing access to 'HistoricalArchives' for free scientific research"^^xsd:string ;
    dc:title "Educational Program on Theater History and Cultural Heritage"^^xsd:string ;
    odrl:assigner drk:DE_Staatstheater_Augsburg ;
    odrl:assignee drk:LocalUniversity ;
    odrl:permission [
        a odrl:Permission ;
        odrl:target drk:HistoricalArchives ;
        odrl:action odrl:use ;
        odrl:refinement [
            a odrl:Constraint ;
            odrl:leftOperand odrl:purpose ;
            odrl:operator odrl:isA ;
            odrl:rightOperand "ScientificResearch"^^xsd:string
        ]
    ] .
# ```

# ### Explanation:
# - **Parties**: Defined DE_Staatstheater_Augsburg and the local university as parties using the `odrl:Party` and `vcard:Organization` classes.
# - **Asset**: Defined `HistoricalArchives` as an asset with a unique identifier and description.
# - **Agreement**: Created an agreement titled "Educational Program on Theater History and Cultural Heritage" with metadata using Dublin Core terms. The agreement includes assigner (DE_Staatstheater_Augsburg) and assignee (LocalUniversity).
# - **Permission**: Granted permission to use `HistoricalArchives` for the purpose of "ScientificResearch".

# This Turtle (TTL) file should be a comprehensive representation of the ODRL policy for the given scenario.
"""
    shacl_ttl_path = r'test\shacl.ttl'
    
    # 检查文件是否存在
    if not os.path.exists(shacl_ttl_path):
        print(f"Error: SHACL file not found at {shacl_ttl_path}")
    else:
        result = validate_odrl_against_shacl(odrl_ttl_data, shacl_ttl_path)
        
        if result is None or None in result:
            print("Validation process aborted due to file parsing errors.")
        else:
            conforms, total_constraints, num_violations, violations = result
            print(f"Total SHACL constraint blocks defined: {total_constraints}")
            print(f"Policy conforms to SHACL shapes: {conforms}")
            print(f"Number of violated constraint paths/blocks: {num_violations}\n")
            print(json.dumps(violations, indent=4))
            
            # if num_violations > 0:
            #     print("Detailed violation reports (grouped by result path):")
            #     # 按路径分组显示
            #     path_groups = {}
            #     for report in violations:
            #         key = report['result_path']
            #         if key not in path_groups:
            #             path_groups[key] = []
            #         path_groups[key].append(report)
                
            #     # 按照路径字符串排序输出，提高可读性
            #     sorted_paths = sorted(path_groups.keys())

            #     for i, path in enumerate(sorted_paths, 1):
            #         reports = path_groups[path]
            #         # 统计受影响的焦点节点数量
            #         affected_focus_nodes = set(r['focus_node'] for r in reports)
                    
            #         print(f"Violation Group #{i}:")
            #         print(f"┌─ Result Path: {path}")
            #         print(f"├─ Source Shape: {reports[0]['source_shape']}") # 假设同一路径的违规来自同一形状
            #         print(f"├─ Affected Nodes Count: {len(affected_focus_nodes)}")
            #         print(f"├─ Total Violation Instances: {len(reports)}")
            #         print(f"└─ Sample Message: {reports[0]['message']}\n")

        #         report = {
        #     'focus_node': str(results_graph.value(result, SH.focusNode)),
        #     'result_path': path_str,
        #     'message': str(results_graph.value(result, SH.resultMessage)),
        #     'constraint_component': str(results_graph.value(result, SH.sourceConstraintComponent)),
        #     'source_shape': str(results_graph.value(result, SH.sourceShape)), # 添加违规来源形状
        #     'value': str(results_graph.value(result, SH.value)) if results_graph.value(result, SH.value) else "N/A"
        # }