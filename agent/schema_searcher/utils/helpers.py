from agent.prompt_builder.core.data_models import SchemaContext

def search_results_to_schema_context(search_results):
    """
    Enhanced converter that handles both intelligent retrieval dict format 
    and traditional list format for backward compatibility.
    FIXED: Handles intelligent orchestration schema context properly.
    """
    
    # CRITICAL FIX: Handle intelligent retrieval dict format
    if isinstance(search_results, dict):
        return _convert_intelligent_schema_dict(search_results)
    
    # Original logic for list format (backward compatibility)
    return _convert_traditional_list_format(search_results)

def _convert_intelligent_schema_dict(schema_dict):
    """
    Convert intelligent retrieval dictionary to SchemaContext.
    This is the NEW function that fixes your 68-character prompt issue.
    """
    tables = {}
    column_details = []
    total_columns = 0
    has_xml_fields = False
    
    # Extract from intelligent schema format
    tables_list = schema_dict.get('tables', [])
    columns_by_table = schema_dict.get('columns_by_table', {})
    xml_columns_by_table = schema_dict.get('xml_columns_by_table', {})
    
    # Process each table and its columns
    for table_name in tables_list:
        table_columns = []
        
        # Get columns for this table
        if table_name in columns_by_table:
            for col_info in columns_by_table[table_name]:
                column_name = col_info.get('column', '')
                column_type = col_info.get('type', 'unknown')
                description = col_info.get('description', '')
                is_xml = col_info.get('is_xml_column', False)
                
                table_columns.append(column_name)
                column_details.append({
                    "table": table_name,
                    "column": column_name,
                    "type": column_type,
                    "datatype": column_type,
                    "description": description,
                    "is_xml": is_xml
                })
                
                if is_xml or (isinstance(column_type, str) and "xml" in column_type.lower()):
                    has_xml_fields = True
                
                total_columns += 1
        
        tables[table_name] = table_columns
    
    # Extract relationships if available
    relationships = []
    table_relationships = schema_dict.get('table_relationships', [])
    if isinstance(table_relationships, list):
        for rel in table_relationships[:10]:  # Limit relationships
            if isinstance(rel, dict):
                relationships.append({
                    'source_table': rel.get('source_table', ''),
                    'source_column': rel.get('source_column', ''),
                    'target_table': rel.get('target_table', ''),
                    'target_column': rel.get('target_column', ''),
                    'join_type': rel.get('join_type', 'inner')
                })
    
    # Extract XML mappings if available
    xml_mappings = []
    if xml_columns_by_table:
        for table, xml_cols in xml_columns_by_table.items():
            for xml_col in xml_cols:
                xml_mappings.append({
                    'table': table,
                    'xml_column': xml_col.get('name', ''),
                    'xpath': xml_col.get('xpath', ''),
                    'data_type': xml_col.get('data_type', '')
                })
    
    total_tables = len(tables)
    
    # Get confidence from intelligent schema
    confidence = schema_dict.get('search_metadata', {}).get('confidence', 0.8)
    confidence_range = (max(0.1, confidence - 0.1), min(1.0, confidence + 0.1))
    
    return SchemaContext(
        tables=tables,
        column_details=column_details,
        relationships=relationships,
        xml_mappings=xml_mappings,
        primary_keys={},  # Could be enhanced later
        foreign_keys=[],  # Could be enhanced later
        total_columns=total_columns,
        total_tables=total_tables,
        has_xml_fields=has_xml_fields,
        confidence_range=confidence_range
    )

def _convert_traditional_list_format(search_results):
    """
    Original logic for backward compatibility with list format.
    """
    tables = {}
    column_details = []
    total_columns = 0
    has_xml_fields = False

    for entry in search_results:
        table = getattr(entry, "table", None) or entry.get("table")
        column = getattr(entry, "column", None) or entry.get("column")
        coltype = getattr(entry, "type", None) or entry.get("type")
        datatype = getattr(entry, "datatype", None) or entry.get("datatype") or None
        description = getattr(entry, "description", "") or entry.get("description", "")

        if table is None or column is None:
            continue

        tables.setdefault(table, []).append(column)
        column_details.append({
            "table": table,
            "column": column,
            "type": coltype,
            "datatype": datatype if datatype is not None else coltype,
            "description": description
        })

        # XML detection logic (your original code)
        _str_coltype = None
        if coltype is not None:
            if isinstance(coltype, str):
                _str_coltype = coltype
            elif hasattr(coltype, "value"):
                _str_coltype = coltype.value
            elif hasattr(coltype, "name"):
                _str_coltype = coltype.name

        if _str_coltype and "xml" in _str_coltype.lower():
            has_xml_fields = True

        total_columns += 1

    total_tables = len(tables)

    return SchemaContext(
        tables=tables,
        column_details=column_details,
        relationships=[],
        xml_mappings=[],
        primary_keys={},
        foreign_keys=[],
        total_columns=total_columns,
        total_tables=total_tables,
        has_xml_fields=has_xml_fields,
        confidence_range=(0.7, 1.0)
    )
