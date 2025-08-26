from agent.prompt_builder.core.data_models import SchemaContext

def search_results_to_schema_context(search_results):
    """
    Converts a list of search results (RetrievedColumn objects, namedtuples, or dicts)
    into a SchemaContext for prompt assembly.
    """
    tables = {}
    column_details = []
    total_columns = 0
    has_xml_fields = False

    for entry in search_results:
        # Unified access for object or dict
        table = getattr(entry, "table", None) or entry.get("table")
        column = getattr(entry, "column", None) or entry.get("column")
        coltype = getattr(entry, "type", None) or entry.get("type")
        description = getattr(entry, "description", "") or entry.get("description", "")

        if table is None or column is None:
            continue

        tables.setdefault(table, []).append(column)
        column_details.append({
            "table": table,
            "column": column,
            "type": coltype,
            "description": description
        })

        # --- Robust xml detection ---
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
