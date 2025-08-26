import logging
import asyncio  # Add this import for async support
from orchestrator.agents.base_agent import BaseAgent

class WorkflowPlanner(BaseAgent):
    """
    WorkflowPlanner: Orchestrates end-to-end query workflows by sequencing schema search,
    prompt building, SQL generation, and execution steps.
    """

    def __init__(self, config=None, logger=None):
        super().__init__(config=config, logger=logger)
        self.logger.info("WorkflowPlanner initialized.")

    async def run_workflow(  # Change to async def
        self,
        user_query,
        schema_searcher,
        search_adapter_func,
        prompt_assembler,
        sql_generator,
        sql_executor,
        prompt_options=None,
    ):
        """
        Execute the full pipeline:
        - schema search
        - adapt to SchemaContext
        - build prompt
        - generate SQL
        - execute SQL

        Returns: result dict with all outputs
        """
        self.log_workflow_start()
        try:
            # 1. Schema Search (assume sync; await if it's async)
            self.logger.info("Running schema search ...")
            search_results = schema_searcher.search(user_query)  # Await if async: await schema_searcher.search(...)
            self.update_state("search_results", search_results)

            # 2. Adapt search results to SchemaContext (sync)
            self.logger.info("Adapting search results to SchemaContext ...")
            schema_context = search_adapter_func(search_results)
            self.update_state("schema_context", schema_context)

            # 3. Build prompt (sync)
            self.logger.info("Building prompt ...")
            structured_prompt = prompt_assembler.assemble_simple_prompt(
                user_query=user_query,
                schema_context=schema_context,
                options=prompt_options
            )
            prompt_text = structured_prompt.get_full_prompt()
            self.update_state("prompt", prompt_text)

            # 4. Generate SQL (key async step)
            self.logger.info("Generating SQL ...")
            sql_query = await sql_generator.generate(prompt_text)  # Add await here
            self.update_state("sql", sql_query)

            # 5. Execute SQL (key async step)
            self.logger.info("Executing SQL ...")
            sql_result = await sql_executor.execute(sql_query)  # Add await here
            self.update_state("sql_result", sql_result)

            self.log_workflow_end()
            # Final output as a dict
            return {
                "user_query": user_query,
                "prompt": prompt_text,
                "sql": sql_query,
                "result": sql_result
            }
        except Exception as e:
            self.handle_error(e, context="Workflow execution")
            return {
                "error": str(e),
                "state": self.state
            }
