import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Embedding
from scipy.stats import kendalltau
from abc import abstractmethod


class MLP(Model):
    def __init__(self, batch_size: int, validation_frequency: int):
        super().__init__()
        self.batch_size = batch_size
        self.validation_frequency = validation_frequency

    @abstractmethod
    def unpack_batch_with_labels(self, batch: tuple[tf.Tensor, ...]):
        pass

    @tf.function
    def train_step(self, batch: tuple[tf.Tensor]):
        call_input, target = self.unpack_batch_with_labels(batch)
        with tf.GradientTape() as tape:
            y_pred = self.__call__(call_input)
            loss_value_true_labels = self.loss_computer_true_labels(target, y_pred)

            # TODO: remember that the goal is to find the top-k, so some samples
            # are more important than others
            loss_value = loss_value_true_labels \
                         + tf.keras.regularizers.L1(l1=1e-6)(self.dense_layer_1.kernel) \
                         + tf.keras.regularizers.L1(l1=1e-6)(self.dense_layer_2.kernel)

        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

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
                label = batch[-1].numpy()
                if len(label.shape) == 2:
                    label = label[:, 1]
                labels.append(label)
            predictions.append(y_pred.numpy())
            id_list.append(batch[0].numpy())
            config_index_list.append(batch[1].numpy())

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
            training_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset):

        self.fit_normalizations(training_dataset)

        iteration = 0
        epoch = 0

        self.validations_without_improvement = 0
        self.best_historic_loss = np.inf

        should_stop = False
        while not should_stop:
            for batch in training_dataset:
                training_loss = self.train_step(batch)
                iteration += 1
                if iteration % 100 == 0:
                    # TODO: use tensorboard -> tune learning rate
                    print(
                        'iteration', iteration,
                        'training loss', training_loss.numpy(),
                        f'lr {self.optimizer.learning_rate.numpy():.5f}')

                if iteration % self.validation_frequency == 0:
                    val_df = self.predict_over_dataset(validation_dataset, return_labels=True)
                    validation_loss = self.compute_validation_loss(val_df)
                    print(f'epoch {epoch}, it {iteration} validation loss {validation_loss:.3f}')
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
    def unpack_batch_with_labels(self, batch: tuple[tf.Tensor, ...]):
        tile_ids, config_indexes, config_descriptors, normalized_runtimes = batch
        return config_descriptors, normalized_runtimes

    def __init__(self, batch_size: int, learning_rate: float):
        super().__init__(batch_size, validation_frequency=10_000)
        self.normalization_layer = Normalization(axis=-1)
        self.dense_layer_1 = Dense(
            100,
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_1',
        )
        self.dense_layer_2 = Dense(
            10,
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
            decay_steps=1_000,
            decay_rate=0.9,
            staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_computer_true_labels = tf.keras.losses.MeanSquaredError()

        self.max_validations_without_improvement = 2

    def call(self, x):
        x = self.normalization_layer(x)
        x = self.dense_layer_1(x)
        x = self.relu_layer(x)
        x = self.dense_layer_2(x)
        x = self.relu_layer(x)
        x = self.dense_layer_3(x)
        x = tf.reshape(x, (-1, ))
        return x

    def inference_from_batch(self, batch: tuple[tf.Tensor]) -> tf.Tensor:
        tile_ids, config_indexes, config_descriptors = batch[:3]
        prediction = self.__call__(config_descriptors)  # TODO: call with @tf.function
        return prediction

    def compute_validation_loss(self, validation_df: pd.DataFrame) -> float:
        predictions = validation_df['prediction']
        targets = validation_df['target']
        validation_score = float(np.mean(np.square(predictions - targets)))
        validation_loss = validation_score
        return validation_loss

    def fit_normalizations(self, dataset):
        config_desc_list = []
        for i, batch in enumerate(dataset):
            tile_ids, config_indexes, config_descriptors, normalized_runtimes = batch
            config_desc_list.append(config_descriptors)
            if i == 100:
                break

        config_desc_list = np.concatenate(config_desc_list, axis=0)
        self.normalization_layer_config_nodes.adapt(config_desc_list)


class LayoutMLP(MLP):
    def __init__(self, batch_size: int, learning_rate: float, mask_max_len: int):
        super().__init__(batch_size, validation_frequency=1_000)
        self.mask_max_len = mask_max_len
        self.normalization_layer_config_nodes = Normalization(axis=-1)
        self.normalization_layer_graph_descriptor = Normalization(axis=-1)
        self.dense_layer_1 = Dense(
            100,
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_1',
        )  # kernel: (18, 100)
        self.dense_layer_2 = Dense(
            25,
            activation=None,
            kernel_initializer='he_uniform',
            name='dense_layer_2',
        )
        self.dense_layer_3 = Dense(
            2,
            name='dense_layer_3'
        )

        self.relu_layer = ReLU(max_value=50.0, negative_slope=0.01)
        self.embedding_layer = Embedding(121, 32, input_length=mask_max_len)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1_000,
            decay_rate=0.9,
            staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_computer_true_labels = tf.keras.losses.MeanAbsoluteError()

        self.max_validations_without_improvement = 10

    @tf.function
    def train_step(self, batch: tuple[tf.Tensor]):
        call_input, targets = self.unpack_batch_with_labels(batch)
        with tf.GradientTape() as tape:
            y_pred = self.__call__(call_input)
            absolute_target = targets[:, 0]
            normalized_target = targets[:, 1]
            loss_from_absolute = self.loss_computer_true_labels(
                absolute_target, y_pred[:, 0])

            loss_from_normalized = self.loss_computer_true_labels(
                normalized_target, y_pred[:, 1])

            loss_value = 0.2 * loss_from_absolute + loss_from_normalized \
                         + tf.keras.regularizers.L1(l1=1e-7)(self.dense_layer_1.kernel) \
                         + tf.keras.regularizers.L1(l1=1e-7)(self.dense_layer_2.kernel)

        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

    def call(self, x):
        config_descriptor, valid_mask, graph_descriptor = x
        node_operations = config_descriptor[:, :, -1]
        config_descriptor = config_descriptor[:, :, :-1]
        node_operations = tf.cast(node_operations, tf.int32)
        # node_operations.shape == (batch_size, mask_max_len)
        node_embedding = self.embedding_layer(node_operations)
        # node_embedding.shape == (batch_size, mask_max_len, embed_len)
        
        x = self.normalization_layer_config_nodes(config_descriptor)
        x = tf.concat([x, node_embedding], axis=-1)
        x = self.dense_layer_1(x)
        x = self.relu_layer(x)  # (batch_size, n_config_nodes_upper_limit, n_units)

        float_mask = tf.sequence_mask(valid_mask, self.mask_max_len, dtype=tf.float32)
        # (batch_size, n_config_nodes_upper_limit)

        float_mask = tf.expand_dims(float_mask, axis=-1)
        x = x * float_mask
        x = tf.reduce_mean(x, axis=1)

        normal_graph_descriptor = self.normalization_layer_graph_descriptor(graph_descriptor)
        x = tf.concat([x, normal_graph_descriptor], axis=-1)
        x = self.dense_layer_2(x)
        x = self.relu_layer(x)
        x = self.dense_layer_3(x)
        return x

    def inference_from_batch(self, batch: tuple[tf.Tensor]) -> tf.Tensor:
        tile_ids, config_indexes, config_descriptors, valid_mask, graph_descriptor = batch[:5]
        prediction = self.__call__((config_descriptors, valid_mask, graph_descriptor))  # TODO: call with @tf.function
        return prediction[:, 1]

    def compute_validation_loss(self, validation_df: pd.DataFrame) -> float:
        assert 'target' in validation_df.columns

        def compute_layout_score_group(df):
            predicted_order = df.sort_values('prediction')['config_index'].values
            true_order = df.sort_values('target')['config_index'].values
            score, _ = kendalltau(predicted_order, true_order)
            return score

        mean = np.mean(validation_df.groupby('ID').apply(compute_layout_score_group))
        return -float(mean)

    def unpack_batch_with_labels(self, batch: tuple[tf.Tensor, ...]):
        tile_ids, config_indexes, config_descriptors, valid_mask, graph_descriptor, normalized_runtimes = batch
        return (config_descriptors, valid_mask, graph_descriptor), normalized_runtimes

    def fit_normalizations(self, dataset):
        config_desc_list = []
        graph_desc_list = []
        for i, batch in enumerate(dataset):
            tile_ids, config_indexes, config_descriptors, valid_mask, graph_descriptor, normalized_runtimes = batch
            config_desc_list.append(config_descriptors[:, :, :-1])  # last feature is not used here (layer type)
            graph_desc_list.append(graph_descriptor)
            if i == 100:
                break

        config_desc_list = np.concatenate(config_desc_list, axis=0)
        self.normalization_layer_config_nodes.adapt(config_desc_list)

        graph_desc_list = np.concatenate(graph_desc_list, axis=0)
        self.normalization_layer_graph_descriptor.adapt(graph_desc_list)
