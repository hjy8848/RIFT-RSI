from usami_harness.service import Experiment,Execution,ServiceSession as _Session,replay_execution as _replay
from . import adapter


class ServiceSession(_Session):
    def __init__(self,experiment,client):super().__init__(adapter,experiment,client)


def replay_execution(execution,experiment=None):return _replay(adapter,execution,experiment)
