import ray
from ray import tune
from ray import train
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from dataset import LayoutDataset
from models import LayoutMLP


def training_function(config: dict):
    learning_rate = config['learning_rate']
    dataset_take = config['dataset_take']
    batch_per_file_size = config['batch_per_file_size']
    node_embedding_size = config['node_embedding_size']
    loss = config['loss']
    l1_multiplier = config['l1_multiplier']
    batch_size = 96
    dataset = LayoutDataset(
        batch_size,
        dataset_take,
        build_tfrecords=False,
        batch_per_file_size=batch_per_file_size
    )

    mlp = LayoutMLP(
        batch_size,
        learning_rate=learning_rate,
        batch_per_file_size=batch_per_file_size,
        node_embedding_size=node_embedding_size,
        validation_frequency=10_000,
        validations_without_improvement=5,
        layer_sizes=[config[f'layer_{i}'] for i in range(4)],
        loss=loss,
        l1_multiplier=l1_multiplier,
        n_siblings=dataset.n_siblings,
        output_name=None
    )
    mlp.train(
        dataset,
        lambda it, val_score: train.report(
            {'iterations': it, 'validation_kendall': val_score}))


if __name__ == '__main__':
    ray.init(configure_logging=False)
    algo = HyperOptSearch(n_initial_points=10)
    scheduler = AsyncHyperBandScheduler(
        time_attr="iterations", grace_period=50_000, max_t=600_000)

    hp_config = {
        "learning_rate": tune.loguniform(5e-4, 5e-3),
        "batch_per_file_size": tune.choice([6, 8, 12]),
        "node_embedding_size": tune.lograndint(8, 20),
        'layer_0': tune.lograndint(96, 160),
        'layer_1': tune.lograndint(48, 96),
        'layer_2': tune.lograndint(16, 48),
        'layer_3': tune.lograndint(16, 48),
        'dataset_take': tune.lograndint(1000, 5000),
        'loss': tune.choice(['pairwise_hinge', 'list_mle']),
        'l1_multiplier': tune.loguniform(1e-10, 1e-6)
    }

    trainable_with_cpu_gpu = tune.with_resources(
        training_function, {"cpu": 16, "gpu": 1})
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
