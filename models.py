import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
import tensorflow_ranking as tfr
from scipy.stats import kendalltau
from abc import abstractmethod, ABC
from typing import List


class MLP(Model, ABC):
    def __init__(self, batch_size: int, validation_frequency: int):
        super().__init__()
        self.batch_size = batch_size
        self.validation_frequency = validation_frequency
        self.debug = False

        self.validations_without_improvement = 0
        self.best_historic_loss = np.inf

    @abstractmethod
    def unpack_batch_with_labels(self, batch: tuple[tf.Tensor, ...]):
        pass

    @abstractmethod
    def inference_from_batch(self, batch: tuple[tf.Tensor]) -> tf.Tensor:
        pass

    def predict_over_dataset(self, dataset: tf.data.Dataset, return_labels) -> pd.DataFrame:
        predictions = []
        if return_labels:
            labels = []
        id_list = []
        config_index_list = []
        for batch in dataset:
            y_pred = self.inference_from_batch(batch)
            if return_labels:
                label = batch['target'].numpy()
                labels.append(label)
            predictions.append(y_pred.numpy())
            id_list.append(batch[self.id_key].numpy())
            config_index_list.append(batch['config_index'].numpy())

        predictions = np.concatenate(predictions, axis=0)
        if return_labels:
            labels = np.concatenate(labels, axis=0)
        id_list = np.concatenate(id_list, axis=0)
        config_index_list = np.concatenate(config_index_list, axis=0)

        if return_labels:
            data = [id_list, config_index_list, predictions, labels]
            columns = ['ID', 'config_index', 'prediction', 'target']
            data_dict = {}
            for name, column_data in zip(columns, data):
                data_dict[name] = pd.Series(column_data)

        else:
            data = [id_list, config_index_list, predictions]
            columns = ['ID', 'config_index', 'prediction']
            data_dict = {}
            for name, column_data in zip(columns, data):
                data_dict[name] = pd.Series(column_data)

        df = pd.DataFrame(data_dict)

        return df

    @abstractmethod
    def fit_normalizations(self, dataset):
        pass

    def train(
            self,
            dataset,
            validation_callback=None):

        training_dataset = dataset.train_data
        validation_dataset = dataset.valid_data
        self.fit_normalizations(training_dataset)

        iteration = 0
        epoch = 0

        should_stop = False
        while not should_stop:
            for batch in training_dataset:
                training_loss = self.train_step(batch)
                iteration += 1
                if iteration % 500 == 0:
                    # TODO: use tensorboard -> tune learning rate
                    print(
                        'iteration', iteration,
                        'training loss', training_loss.numpy(),
                        f'lr {self.optimizer.learning_rate.numpy():.5f}')

                if self.debug and iteration == 150:
                    tf.profiler.experimental.start('tf_profile_dir')
                if self.debug and iteration == 250:
                    tf.profiler.experimental.stop()
                    return

                if iteration % self.validation_frequency == 0:
                    val_df = self.predict_over_dataset(
                        validation_dataset, return_labels=True)
                    validation_loss = self.compute_validation_loss(val_df)
                    print(f'epoch {epoch}, it {iteration} validation loss {validation_loss:.3f}')
                    if validation_callback is not None:
                        validation_callback(iteration, -validation_loss)
                    should_stop = self._evaluate_training_stopper(validation_loss)
                    if should_stop:
                        print('stopping training')
                        break
            epoch += 1

    @abstractmethod
    def compute_validation_loss(self, validation_df: pd.DataFrame) -> float:
        pass

    def _evaluate_training_stopper(self, current_validation_loss):
        if current_validation_loss < self.best_historic_loss:
            self.best_historic_loss = current_validation_loss
            self.validations_without_improvement = 0
            return False

        self.validations_without_improvement += 1
        if self.validations_without_improvement >= self.max_validations_without_improvement:
            return True
        else:
            return False


class TileMLP(MLP):
    def unpack_batch_with_labels(self, batch: dict[str, tf.Tensor]):
        config_descriptors = batch['config_descriptor']
        graph_descriptors = batch['graph_descriptor']
        normalized_runtimes = batch['target']
        return (config_descriptors, graph_descriptors), normalized_runtimes

    def __init__(
            self,
            batch_size: int,
            learning_rate: float,
            batch_per_file_size: int
    ):
        super().__init__(batch_size, validation_frequency=10_000)
        self.id_key = 'tile_id'
        self.batch_per_file_size = batch_per_file_size
        self.dense_layer_1 = Dense(
            250,
            name='dense_layer_1',
        )
        self.dropout = Dropout(0.15)
        self.dense_layer_2 = Dense(
            100,
            name='dense_layer_2',
        )
        self.dense_layer_3 = Dense(
            1,
            name='dense_layer_3',
            use_bias=False
        )

        self.activation = tf.nn.silu

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=100000,
            warmup_target=learning_rate,
            warmup_steps=5000,
            alpha=5e-2
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        self.loss_computer = tfr.keras.losses.PairwiseHingeLoss()

        self.max_validations_without_improvement = 5

    def call(self, x, training=False):
        x, graph_descriptor = x

        x = tf.clip_by_value(
            x,
            clip_value_min=1 / np.e,
            clip_value_max=1e6
        )
        x = tf.math.log(x)

        x = (x - self.mean) / self.std
        x = tf.concat([x, graph_descriptor], axis=1)

        x = self.dense_layer_1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.dense_layer_2(x)
        x = self.activation(x)
        x = self.dense_layer_3(x)
        x = tf.reshape(x, (-1,))
        return x

    @tf.function
    def train_step(self, batch: dict[str, tf.Tensor]):
        call_input, target = self.unpack_batch_with_labels(batch)
        with tf.GradientTape() as tape:
            y_pred = self.__call__(call_input, training=True)
            loss_value = self.loss_computer(
                tf.reshape(target, (-1, self.batch_per_file_size)),
                tf.reshape(y_pred, (-1, self.batch_per_file_size))
            )

        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

    @tf.function
    def inference_from_batch(self, batch: dict[str, tf.Tensor]) -> tf.Tensor:
        config_descriptor = batch['config_descriptor']
        graph_descriptor = batch['graph_descriptor']
        prediction = self.__call__((config_descriptor, graph_descriptor))
        return prediction

    def compute_validation_loss(self, validation_df: pd.DataFrame) -> float:
        def metric_per_id(df):
            top = df.sort_values('prediction').iloc[:5]
            best_attempt = np.min(top['target'])
            best_target = np.min(df['target'])
            return 2 - np.exp(best_attempt - best_target)

        metrics = validation_df.groupby('ID').apply(metric_per_id)
        return -np.mean(metrics)

    def fit_normalizations(self, dataset):
        config_desc_list = []
        for i, batch in enumerate(dataset):
            config_descriptor = batch['config_descriptor']
            config_desc_list.append(config_descriptor)
            if i == 100:
                break

        config_desc_list = np.concatenate(config_desc_list, axis=0)
        config_desc_list = np.clip(
            config_desc_list,
            a_min=1 / np.e,
            a_max=1e10
        )
        config_desc_list = np.log(config_desc_list)
        mean = np.mean(config_desc_list, axis=0)
        std = np.std(config_desc_list, axis=0)
        std = np.clip(std, 1.0, None)
        self.mean = mean
        self.std = std


class LayoutMLP(MLP):
    def __init__(
            self,
            batch_size: int,
            learning_rate: float,
            mask_max_len: int,
            validation_frequency: int,
            batch_per_file_size: int,
            layer_sizes: List[int],
            validations_without_improvement: int,
            node_embedding_size: int,
            loss: str,
            l1_multiplier: float,
            n_siblings: int
    ):
        super().__init__(batch_size, validation_frequency=validation_frequency)
        self.id_key = 'layout_id'
        self.mask_max_len = mask_max_len
        self.batch_per_file_size = batch_per_file_size

        self.dropout = Dropout(0.1, noise_shape=(batch_size, 1, layer_sizes[1]))
        self.dense_layer_node_1 = Dense(
            layer_sizes[0],
            name='dense_layer_node_1',
        )
        self.dense_layer_node_2 = Dense(
            layer_sizes[1],
            name='dense_layer_node_2',
        )

        self.dense_layer_global_1 = Dense(
            layer_sizes[2],
            name='dense_layer_global_1',
        )
        self.dense_layer_global_2 = Dense(
            layer_sizes[3],
            name='dense_layer_global_2',
        )

        self.dense_layer_global_3 = Dense(
            1,
            name='dense_layer_global_3',
            use_bias=False
        )

        self.activation = tf.nn.silu
        self.node_embedding_size = node_embedding_size
        self.embedding_layer_node_ops = Embedding(
            121, node_embedding_size, input_length=mask_max_len)

        self.text_vectorization = tf.keras.layers.TextVectorization(
            standardize=None,
            split=None,
            output_mode='int',
            vocabulary=[
                b'layoutnlpdefault',
                b'layoutnlprandom',
                b'layoutxladefault',
                b'layoutxlarandom'
            ]
        )
        self.embedding_layer_subset_info = Embedding(6, 6, input_length=1)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=250000,
            warmup_target=learning_rate,
            warmup_steps=10000,
            alpha=1e-2
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=10.0
        )
        if loss == 'pairwise_hinge':
            self.loss_computer = tfr.keras.losses.PairwiseHingeLoss()
        elif loss == 'list_mle':
            self.loss_computer = tfr.keras.losses.ListMLELoss()
        else:
            raise ValueError(f'{loss} is not a valid loss')

        self.l1_multiplier = l1_multiplier
        self.max_validations_without_improvement = validations_without_improvement
        self.n_siblings = n_siblings

        self.sibling_output_shapes = np.arange(self.n_siblings*6)
        self.node_output_shapes = np.arange(6) + self.n_siblings*6+self.n_siblings*(18+1)
        self.parents_output_shapes = np.arange(12) + self.n_siblings*6+self.n_siblings*(18+1)+6+6+4+4+6+18

        self.node_order = np.arange(6) + self.n_siblings*6+self.n_siblings*(18+1)+6+6+4+4+6

        self.siblings_order = []
        for sibling_i in range(self.n_siblings):
            sibling_order = np.arange(6) + self.n_siblings*6 + sibling_i*19
            self.siblings_order.append(sibling_order)
        self.siblings_order = np.concatenate(self.siblings_order)

        self.parents_order = [
            np.arange(6) + self.n_siblings*6+self.n_siblings*(18+1)+6+6+4+4+6 + 6,
            np.arange(6) + self.n_siblings*6+self.n_siblings*(18+1)+6+6+4+4+6 + 12,
        ]
        self.parents_order = np.concatenate(self.parents_order)

        self.features_with_dims = np.concatenate([
            self.sibling_output_shapes,
            self.node_output_shapes,
            self.parents_output_shapes
        ])
        self.features_with_dims_complement = np.array([
            i for i in range(149-(1+2+self.n_siblings)) if i not in self.features_with_dims])

    @tf.function
    def train_step(self, batch: dict[str, tf.Tensor]):
        call_input, targets = self.unpack_batch_with_labels(batch)
        with tf.GradientTape() as tape:
            y_pred = self.__call__(call_input, training=True)
            loss_value = self.loss_computer(
                tf.reshape(targets, (-1, self.batch_per_file_size)),
                tf.reshape(y_pred, (-1, self.batch_per_file_size))
            )

            loss_value = loss_value \
                             + tf.keras.regularizers.L1(l1=self.l1_multiplier)(self.dense_layer_node_1.kernel)

        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

    def call(self, x, training=False):
        layout_ids, node_descriptor, valid_mask, graph_descriptor = x

        with tf.device('/cpu:0'):
            subset_info = tf.map_fn(
                lambda layout_id: tf.strings.reduce_join(
                    tf.strings.split(layout_id, ":")[:3]),
                layout_ids
            )

        subset_info = self.text_vectorization(subset_info)
        subset_info = tf.expand_dims(subset_info, axis=-1)
        subset_info = self.embedding_layer_subset_info(subset_info)
        subset_info = subset_info[:, 0, :]

        n_opcodes = 1 + 2 + self.n_siblings
        node_operations = node_descriptor[:, :, -n_opcodes:]
        node_descriptor = node_descriptor[:, :, :-n_opcodes]
        node_operations = tf.cast(node_operations, tf.int32)
        # node_operations.shape == (batch_size, mask_max_len, (1+2+self.n_siblings))
        node_embedding = self.embedding_layer_node_ops(node_operations)
        # node_embedding.shape == (batch_size, mask_max_len, (1+2+self.n_siblings), embed_len)
        node_embedding = tf.reshape(
            node_embedding,
            (-1, self.mask_max_len, n_opcodes*self.node_embedding_size))

        node_descriptor_wo_shapes = tf.gather(
            node_descriptor,
            self.features_with_dims_complement,
            axis=-1)

        node_descriptor_shapes = tf.gather(
            node_descriptor,
            self.features_with_dims,
            axis=-1)

        new_order = tf.concat([
            self.siblings_order,
            self.node_order,
            self.parents_order
        ], axis=0)
        new_order = tf.gather(
            node_descriptor,
            new_order,
            axis=-1
        )
        new_order = tf.cast(new_order, tf.int32)

        reordered_shape_list = []
        for i in range(1+2+self.n_siblings):
            reordered_shapes = tf.gather(
                node_descriptor_shapes[:, :, i*6:(i+1)*6],
                new_order[:, :, i*6:(i+1)*6],
                axis=2,
                batch_dims=2
            )
            reordered_shape_list.append(reordered_shapes)
        reordered_shapes = tf.concat(reordered_shape_list, axis=-1)

        node_descriptor_shapes = tf.concat([
            node_descriptor_shapes,
            reordered_shapes
        ], axis=-1)

        node_descriptor_shapes = tf.clip_by_value(
            node_descriptor_shapes,
            clip_value_min=1/np.e,
            clip_value_max=1e10
        )
        node_descriptor_shapes = tf.math.log(node_descriptor_shapes)

        node_descriptor = tf.concat([
            node_descriptor_shapes,
            node_descriptor_wo_shapes
        ], axis=-1)

        x = (node_descriptor - self.mean) / self.std

        x = tf.concat([x, node_embedding], axis=-1)

        x = self.dense_layer_node_1(x)
        x = self.activation(x)

        x = self.dense_layer_node_2(x)
        x = self.activation(x)  # (batch_size, n_config_nodes_upper_limit, n_units)
        x = self.dropout(x, training=training)

        float_mask = tf.sequence_mask(valid_mask, self.mask_max_len, dtype=tf.float32)
        # (batch_size, n_config_nodes_upper_limit)

        float_mask = tf.expand_dims(float_mask, axis=-1)
        x = x * float_mask

        x = tf.reduce_sum(x, axis=1)
        x = x / tf.expand_dims(tf.cast(valid_mask, tf.float32), axis=-1)

        x = tf.concat([x, graph_descriptor, subset_info], axis=-1)
        x = tf.concat([x, graph_descriptor], axis=-1)
        x = self.dense_layer_global_1(x)
        x = self.activation(x)
        x = self.dense_layer_global_2(x)
        x = self.activation(x)
        x = self.dense_layer_global_3(x)
        x = tf.reshape(x, (-1,))
        return x

    @tf.function
    def inference_from_batch(self, batch: dict[str, tf.Tensor]) -> tf.Tensor:
        layout_ids = batch['layout_id']
        config_descriptors = batch['node_descriptor']
        valid_mask = batch['valid_nodes']
        graph_descriptor = batch['graph_descriptor']
        prediction = self.__call__((layout_ids, config_descriptors, valid_mask, graph_descriptor))
        return prediction

    def compute_validation_loss(self, validation_df: pd.DataFrame) -> float:
        assert 'target' in validation_df.columns

        def get_set_name_from_id(bin_id):
            x = bin_id.decode('UTF-8')
            x = x.split(':')
            x = x[:3]
            x = ':'.join(x)
            return x

        validation_df['subset'] = validation_df['ID'].map(get_set_name_from_id)

        def compute_layout_score_group(df):
            score, _ = kendalltau(df['prediction'], df['target'])
            return score

        set_means = []
        subsets = [
            'layout:nlp:random',
            'layout:nlp:default',
            'layout:xla:random',
            'layout:xla:default',
        ]
        for subset in subsets:
            val_subset = validation_df[validation_df['subset'] == subset]
            if len(val_subset) == 0:
                continue
            mean = np.mean(val_subset.groupby('ID').apply(compute_layout_score_group))
            print(subset, mean)
            set_means.append(mean)

        return -float(np.mean(set_means))

    def unpack_batch_with_labels(self, batch: dict[str, tf.Tensor]):
        layout_ids = batch['layout_id']
        config_descriptors = batch['node_descriptor']
        valid_mask = batch['valid_nodes']
        graph_descriptor = batch['graph_descriptor']
        target = batch['target']
        return (layout_ids, config_descriptors, valid_mask, graph_descriptor), target

    def fit_normalizations(self, dataset):
        config_desc_list = []
        for i, batch in enumerate(dataset):
            if i % 200 == 0:
                config_descriptors = batch['node_descriptor']
                valid_nodes = batch['valid_nodes']  # ignore zero padding
                node_descriptor = config_descriptors[:, :, :-(1+2+self.n_siblings)].numpy()

                node_descriptor_wo_shapes = node_descriptor[:, :, self.features_with_dims_complement]
                node_descriptor_shapes = node_descriptor[:, :, self.features_with_dims]

                new_order = np.concatenate([
                    self.siblings_order,
                    self.node_order,
                    self.parents_order
                ], axis=0)
                new_order = node_descriptor[:, :, new_order].astype(int)

                reordered_shape_list = []
                for i in range(1 + 2 + self.n_siblings):
                    reordered_shapes = tf.gather(
                        node_descriptor_shapes[:, :, i * 6:(i + 1) * 6],
                        new_order[:, :, i * 6:(i + 1) * 6],
                        axis=2,
                        batch_dims=2
                    ).numpy()
                    reordered_shape_list.append(reordered_shapes)
                reordered_shapes = np.concatenate(reordered_shape_list, axis=-1)

                node_descriptor_shapes = np.concatenate([
                    node_descriptor_shapes,
                    reordered_shapes
                ], axis=-1)

                node_descriptor_shapes = np.clip(
                    node_descriptor_shapes,
                    a_min=1 / np.e,
                    a_max=1e10
                )
                node_descriptor_shapes = np.log(node_descriptor_shapes)

                node_descriptor = np.concatenate([
                    node_descriptor_shapes,
                    node_descriptor_wo_shapes
                ], axis=-1)

                for j in range(self.batch_size):
                    descriptor = node_descriptor[j, :valid_nodes[j], :]
                    config_desc_list.append(descriptor)

            if i > 2_000:
                break

        config_desc_list = np.concatenate(config_desc_list, axis=0)
        mean = np.mean(config_desc_list, axis=0)
        std = np.std(config_desc_list, axis=0)
        std = np.clip(std, 1.0, None)
        self.mean = mean
        self.std = std
