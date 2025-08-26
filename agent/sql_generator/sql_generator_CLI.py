#!/usr/bin/env python
"""
SQL Generator CLI – isolate & debug your SQLGenerator
Usage:
  # Interactive mode
  python sql_generator_cli.py --interactive

  # Batch mode (JSON array of { "query": ..., "context": ... })
  python sql_generator_cli.py --batch --input queries.json --schema schema.json --output results.json
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ✅ CORRECTED IMPORTS - Fixed the module paths
from .generator import SQLGenerator
from .models.request_models import GenerationRequest, QueryContext, UserIntent
from .config import get_config
from .utils.logger import setup_logging

class SQLGeneratorCLI:
    def __init__(self, config):
        setup_logging(config.log_level)
        self.generator = SQLGenerator(config)

    async def run_interactive(self):
        print("🚀 SQL Generator Interactive Mode")
        print("Type your natural-language query and hit ENTER. Type 'exit' to quit.\n")
        while True:
            user_input = input("nl> ").strip()
            if not user_input or user_input.lower() in ("exit","quit"):
                break

            # ✅ CORRECTED: Use the actual API structure from your SQLGenerator
            req = GenerationRequest(
                query_text=user_input,                    # Based on your actual API
                max_complexity=75,                        # Default complexity
                enable_xml_optimization=True,             # Enable XML features
                enable_caching=True,                      # Enable caching
                context=QueryContext(                     # Context structure
                    database_name=None,
                    schema_name='dbo',
                    tables_hint=[],
                    xml_columns=[],
                    user_preferences={},
                    session_id=None,
                    user_intent=None
                )
            )

            try:
                response = await self.generator.generate_sql(req)  # ✅ Correct method name
                print(f"\nSQL> {response.sql_query}")
                print(f"Confidence: {response.confidence.percentage}%")
                if response.warnings:
                    print(f"Warnings: {response.warnings}")
                print()
            except Exception as e:
                print(f"[ERROR] {e}\n")

    async def run_batch(
        self,
        queries_file: Path,
        schema_file: Optional[Path],
        output_file: Optional[Path]
    ):
        # load queries
        items = json.loads(queries_file.read_text())
        schema = {}
        if schema_file:
            schema = json.loads(schema_file.read_text())

        results: List[Dict[str, Any]] = []
        for entry in items:
            q = entry.get("query")
            ctx = entry.get("context", {})
            
            # ✅ CORRECTED: Use proper GenerationRequest structure
            req = GenerationRequest(
                query_text=q,
                max_complexity=entry.get("max_complexity", 75),
                enable_xml_optimization=entry.get("enable_xml", True),
                enable_caching=entry.get("enable_caching", True),
                context=QueryContext(
                    database_name=ctx.get("database_name"),
                    schema_name=ctx.get("schema_name", "dbo"),
                    tables_hint=ctx.get("tables_hint", []),
                    xml_columns=ctx.get("xml_columns", []),
                    user_preferences=ctx.get("user_preferences", {}),
                    session_id=ctx.get("session_id"),
                    user_intent=ctx.get("user_intent")
                )
            )
            
            try:
                response = await self.generator.generate_sql(req)
                results.append({
                    "query": q, 
                    "sql": response.sql_query,
                    "confidence": response.confidence.percentage,
                    "valid": response.is_valid,
                    "warnings": response.warnings
                })
            except Exception as e:
                results.append({"query": q, "error": str(e)})

        out = json.dumps(results, indent=2)
        if output_file:
            output_file.write_text(out)
            print(f"Wrote batch results to {output_file}")
        else:
            print(out)

def parse_args():
    p = argparse.ArgumentParser(description="SQLGenerator CLI")
    p.add_argument("--interactive", action="store_true", help="Run in REPL mode")
    p.add_argument("--batch", action="store_true", help="Run in batch mode")
    p.add_argument("--input", "-i", type=Path, help="Path to JSON file with queries")
    p.add_argument("--schema", "-s", type=Path, help="Path to JSON file with schema context")
    p.add_argument("--output", "-o", type=Path, help="Where to write batch results")
    return p.parse_args()

async def main():
    args = parse_args()
    config = get_config()
    cli = SQLGeneratorCLI(config)

    if args.interactive:
        await cli.run_interactive()
    elif args.batch:
        if not args.input:
            print("Error: --batch requires --input")
            sys.exit(1)
        await cli.run_batch(args.input, args.schema, args.output)
    else:
        print("Error: must supply --interactive or --batch")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
