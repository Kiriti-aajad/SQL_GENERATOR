import logging
import datetime

class BaseAgent:
    """
    Production-ready base agent for orchestrator.
    Provides state management, error handling, and logging for workflow agents.
    """

    def __init__(self, config=None, logger=None):
        """
        Initialize the agent with optional config and logger.
        """
        self.config = config or {}
        self.state = {}
        if logger:
            self.logger = logger
        else:
            # Set up default logger for the agent
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def execute(self, *args, **kwargs):
        """
        Main agent workflow step.
        Must be implemented in child agents.
        """
        raise NotImplementedError("Subclasses of BaseAgent must implement an execute() method.")

    def update_state(self, key, value):
        """
        Track the value of a given step or artifact in the workflow.
        """
        self.state[key] = value
        self.logger.debug(f"State updated: {key} = {repr(value)}")

    def get_state(self, key, default=None):
        """
        Retrieve a state value by key for this agent.
        """
        value = self.state.get(key, default)
        self.logger.debug(f"State retrieved: {key} = {repr(value)}")
        return value

    def handle_error(self, error, context=None, re_raise=True):
        """
        Centralized error handler.
        Logs error with context; can optionally re-raise or suppress.
        """
        context_str = f"Context: {context}" if context else ""
        self.logger.error(f"Error encountered: {str(error)} {context_str}")
        if re_raise:
            raise

    def log_info(self, message):
        """
        Info-level workflow logging.
        """
        self.logger.info(message)

    def log_debug(self, message):
        """
        Debug-level workflow logging.
        """
        self.logger.debug(message)

    def log_workflow_start(self):
        """
        Log the start time of the workflow.
        """
        ts = datetime.datetime.now().isoformat()
        self.logger.info(f"Workflow started at {ts}")
        self.update_state("workflow_start", ts)

    def log_workflow_end(self):
        """
        Log the end time of the workflow.
        """
        ts = datetime.datetime.now().isoformat()
        self.logger.info(f"Workflow finished at {ts}")
        self.update_state("workflow_end", ts)
