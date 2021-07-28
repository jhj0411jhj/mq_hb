from resnet_obj import mf_objective_func_gpu
from resnet_model import ResNetClassifier

test_config = ResNetClassifier.get_hyperparameter_search_space().get_default_configuration()

extra_conf = dict(initial_run=True)
mf_objective_func_gpu(config=test_config, n_resource=27, extra_conf=extra_conf,
                      device='cuda:1', total_resource=81, eta=3)

extra_conf = dict(initial_run=False)
mf_objective_func_gpu(config=test_config, n_resource=81, extra_conf=extra_conf,
                      device='cuda:2', total_resource=81, eta=3)
