import ray
from ray import tune
from ray import train
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from dataset import LayoutDataset
from models import LayoutMLP


def training_function(config: dict):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    batch_per_file_size = config['batch_per_file_size']
    decay_rate = config['decay_rate']
    node_embedding_size = config['node_embedding_size']
    loss = config['loss']
    l1_multiplier = config['l1_multiplier']
    dataset = LayoutDataset(
        batch_size, train_sample_fraction=1.0,
        subset=None, build_tfrecords=False,
        batch_per_file_size=batch_per_file_size
    )

    mlp = LayoutMLP(
        batch_size,
        learning_rate=learning_rate,
        mask_max_len=dataset.n_config_nodes_upper_limit,
        batch_per_file_size=batch_per_file_size,
        decay_rate=decay_rate,
        node_embedding_size=node_embedding_size,
        validation_frequency=15_000,
        validations_without_improvement=3,
        layer_sizes=[config[f'layer_{i}'] for i in range(4)],
        loss=loss,
        l1_multiplier=l1_multiplier
    )
    mlp.train(
        dataset,
        lambda it, val_score: train.report(
            {'iterations': it, 'validation_kendall': val_score}))


if __name__ == '__main__':
    ray.init(configure_logging=False)
    algo = HyperOptSearch(n_initial_points=10)
    scheduler = AsyncHyperBandScheduler(
        time_attr="iterations", grace_period=40_000, max_t=600_000)

    hp_config = {
        "batch_size": 128,
        "learning_rate": tune.loguniform(5e-5, 5e-3),
        "batch_per_file_size": tune.choice([2, 4, 8, 16]),
        "decay_rate": tune.uniform(0.9, 0.99),
        "node_embedding_size": tune.lograndint(8, 64),
        'layer_0': tune.lograndint(64, 256),
        'layer_1': tune.lograndint(64, 256),
        'layer_2': tune.lograndint(64, 256),
        'layer_3': tune.lograndint(32, 128),
        'loss': tune.choice(['pairwise_hinge', 'list_mle']),
        'l1_multiplier': tune.loguniform(1e-9, 1e-6)
    }

    trainable_with_cpu_gpu = tune.with_resources(
        training_function, {"cpu": 14, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        tune_config=tune.TuneConfig(
            metric="validation_kendall",
            mode="max",
            search_alg=algo,
            num_samples=200,
            scheduler=scheduler
        ),
        param_space=hp_config,
    )
    results = tuner.fit()
