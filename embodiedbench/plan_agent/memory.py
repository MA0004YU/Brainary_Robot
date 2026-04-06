class BaseMemory:
    def retrieve_long_term(self, query: str):
        raise NotImplementedError

    def store_long_term(self, case_summary: dict):
        raise NotImplementedError

    def read_working(self, task_id: str):
        raise NotImplementedError

    def write_working(self, task_id: str, state: dict):
        raise NotImplementedError


class NoopMemory(BaseMemory):
    def retrieve_long_term(self, query: str):
        return []

    def store_long_term(self, case_summary: dict):
        return None

    def read_working(self, task_id: str):
        return {}

    def write_working(self, task_id: str, state: dict):
        return None