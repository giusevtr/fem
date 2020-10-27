import sys
from datasets.domain import Domain
from Util.util import get_dummy_data2
from Util.qm import QueryManager

class TestQM:
    @classmethod
    def setup_class(self):
        self.domain = Domain(('A', 'B', 'C', 'D'), [3, 3, 3])
        self.workloads = [('A', 'C')]
        self.query_manager = QueryManager(self.domain, self.workloads)
        self.data = get_dummy_data2(self.domain, 50, self.query_manager)

    def test_query_workload(self):
        assert True