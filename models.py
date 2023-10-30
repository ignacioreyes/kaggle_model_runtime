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
        with tf.device('/cpu:0'):
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
        normalized_runtimes = batch['target']
        return config_descriptors, normalized_runtimes

    def __init__(
            self,
            batch_size: int,
            learning_rate: float,
            batch_per_file_size: int
    ):
        super().__init__(batch_size, validation_frequency=10_000)
        self.id_key = 'tile_id'
        self.batch_per_file_size = batch_per_file_size
        self.normalization_layer = Normalization(axis=-1)
        self.dense_layer_1 = Dense(
            100,
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_1',
        )
        self.dense_layer_2 = Dense(
            30,
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_2',
        )
        self.dense_layer_3 = Dense(
            1,
            name='dense_layer_3'
        )

        self.relu_layer = ReLU(max_value=10.0, negative_slope=0.01)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=5_000,
            decay_rate=0.9,
            staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_computer = tfr.keras.losses.PairwiseHingeLoss()

        self.max_validations_without_improvement = 3

    def call(self, x):
        x = self.normalization_layer(x)
        x = self.dense_layer_1(x)
        x = self.relu_layer(x)
        x = self.dense_layer_2(x)
        x = self.relu_layer(x)
        x = self.dense_layer_3(x)
        x = tf.reshape(x, (-1,))
        return x

    @tf.function
    def train_step(self, batch: dict[str, tf.Tensor]):
        call_input, target = self.unpack_batch_with_labels(batch)
        with tf.GradientTape() as tape:
            y_pred = self.__call__(call_input)
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
        prediction = self.__call__(config_descriptor)
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
        self.normalization_layer.adapt(config_desc_list)


class LayoutMLP(MLP):
    def __init__(
            self,
            batch_size: int,
            learning_rate: float,
            mask_max_len: int,
            validation_frequency: int,
            batch_per_file_size: int,
            layer_sizes: List[int],
            decay_rate: float,
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
        self.k = 12
        self.e = 24
        self.v = layer_sizes[0]
        self.q_layer = Dense(
            self.k * self.e,
            name='q_layer',
        )
        self.k_layer = Dense(
            self.k * self.e,
            name='k_layer',
        )
        self.v_layer = Dense(
            self.k * self.v,
            name='v_layer',
        )
        self.charlie_dropout = Dropout(0.15)
        self.dense_layer_node_2 = Dense(
            layer_sizes[1],
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_node_2',
        )

        self.dense_layer_global_1 = Dense(
            layer_sizes[2],
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_global_1',
        )
        self.dense_layer_global_2 = Dense(
            layer_sizes[3],
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_global_2',
        )

        self.dense_layer_global_3 = Dense(
            1,
            name='dense_layer_global_3',
            use_bias=False
        )

        self.relu_layer = ReLU(negative_slope=0.01)
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

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=learning_rate,
        #     decay_steps=10_000,
        #     decay_rate=decay_rate,
        #     staircase=True)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=250000,
            warmup_target=learning_rate,
            warmup_steps=10000,
            alpha=1e-2
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
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
        self.features_with_dims = np.concatenate([
            np.arange(self.n_siblings*6),
            np.arange(6) + self.n_siblings*6+self.n_siblings*(18+1),
            np.arange(12) + self.n_siblings*6+self.n_siblings*(18+1)+6+6+4+4+6+18
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

            #abs_error = tf.math.abs(targets - y_pred)
            #abs_error = tf.reduce_mean(tf.clip_by_value(abs_error, clip_value_min=0, clip_value_max=5))

            loss_value = loss_value \
                             + tf.keras.regularizers.L1(l1=10*self.l1_multiplier)(self.q_layer.kernel) \
                         + tf.keras.regularizers.L1(l1=10*self.l1_multiplier)(self.k_layer.kernel) \
                         + tf.keras.regularizers.L1(l1=self.l1_multiplier)(self.v_layer.kernel) \
                         + tf.keras.regularizers.L1(l1=self.l1_multiplier)(self.dense_layer_node_2.kernel) \
                         + tf.keras.regularizers.L1(l1=self.l1_multiplier)(self.dense_layer_global_1.kernel) \
                         + tf.keras.regularizers.L1(l1=self.l1_multiplier)(self.dense_layer_global_2.kernel)

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

        # charlie puth
        n = tf.shape(x)[1]
        q_tensor = self.q_layer(x)
        q_tensor = tf.reshape(q_tensor, (-1, n, self.k, self.e))

        k_tensor = self.k_layer(x)
        k_tensor = tf.reshape(k_tensor, (-1, n, self.k, self.e))

        m_tensor = tf.reduce_sum(q_tensor*k_tensor, axis=3)
        s_tensor = tf.nn.softmax(m_tensor/np.sqrt(self.e), axis=-1)
        s_tensor = self.charlie_dropout(s_tensor, training=training)

        v_tensor = self.v_layer(x)
        v_tensor = tf.reshape(v_tensor, (-1, n, self.k, self.v))

        o = v_tensor * tf.expand_dims(s_tensor, axis=-1)
        o = self.relu_layer(o)
        x = tf.reduce_sum(o, axis=2)

        #####

        x = self.dense_layer_node_2(x)
        x = self.relu_layer(x)  # (batch_size, n_config_nodes_upper_limit, n_units)

        float_mask = tf.sequence_mask(valid_mask, self.mask_max_len, dtype=tf.float32)
        # (batch_size, n_config_nodes_upper_limit)

        float_mask = tf.expand_dims(float_mask, axis=-1)
        x = x * float_mask

        x = tf.reduce_sum(x, axis=1)
        x = x / tf.expand_dims(tf.cast(valid_mask, tf.float32), axis=-1)

        x = tf.concat([x, graph_descriptor, subset_info], axis=-1)
        x = tf.concat([x, graph_descriptor], axis=-1)
        x = self.dense_layer_global_1(x)
        x = self.relu_layer(x)
        x = self.dense_layer_global_2(x)
        x = self.relu_layer(x)
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
            if i % 100 == 0:
                config_descriptors = batch['node_descriptor']
                valid_nodes = batch['valid_nodes']  # ignore zero padding
                node_descriptor = config_descriptors[:, :, :-(1+2+self.n_siblings)].numpy()

                node_descriptor_wo_shapes = node_descriptor[:, :, self.features_with_dims_complement]
                node_descriptor_shapes = node_descriptor[:, :, self.features_with_dims]

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

            if i > 1_000:
                break

        config_desc_list = np.concatenate(config_desc_list, axis=0)
        mean = np.mean(config_desc_list, axis=0)
        std = np.std(config_desc_list, axis=0)
        std = np.clip(std, 1.0, None)
        self.mean = mean
        self.std = std
