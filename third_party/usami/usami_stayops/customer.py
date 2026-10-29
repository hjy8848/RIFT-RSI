from usami_harness.customer import Interaction,CustomerAction,CustomerSession as _Session,replay_interaction as _replay
from . import adapter


class CustomerSession(_Session):
    def __init__(self,experiment,customer_client,service_client,method='adaptive'):
        super().__init__(adapter,experiment,customer_client,service_client,method)


def replay_interaction(interaction,experiment=None):return _replay(adapter,interaction,experiment)
