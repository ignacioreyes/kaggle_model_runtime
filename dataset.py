import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from typing import List


def magic_log(x):
    if x <= 0:
        return -1
    return np.log(x)


magic_log_v = np.vectorize(magic_log)


class TileDataset:
    def __init__(self, batch_size: int):
        self.root_dir = 'predict-ai-model-runtime/npz_all/npz/tile/xla'
        self.batch_size = batch_size
        with tf.device('/cpu:0'):
            self.train_data = self._load_set('train')
            self.valid_data = self._load_set('valid')
            self.test_data = self._load_set('test')

    def _load_set(self, set_name: str) -> tf.data.Dataset:
        set_dir = os.path.join(self.root_dir, set_name)
        tile_ids = []
        config_indexes = []
        config_descriptors = []
        normalized_runtimes = []

        for filename in tqdm(os.listdir(set_dir), desc=f'loading {set_name}'):
            tile_id = 'tile:xla:' + filename[:-len('.npz')]
            tile_dict = dict(np.load(os.path.join(set_dir, filename)))
            tile_dict['ID'] = tile_id
            if set_name != 'test':
                target = tile_dict['config_runtime'] / tile_dict['config_runtime_normalizers']
                target = np.log(target)
                tile_dict['normalized_runtime'] = target

            for config_idx in range(len(tile_dict['config_feat'])):
                config_descriptor = tile_dict['config_feat'][config_idx].astype(np.float32)
                if set_name != 'test':
                    normalized_runtime = tile_dict['normalized_runtime'][config_idx].astype(np.float32)
                    normalized_runtimes.append(normalized_runtime)

                tile_ids.append(tile_id)
                config_indexes.append(config_idx)
                config_descriptors.append(config_descriptor)

        tile_ids = np.array(tile_ids)
        config_indexes = np.array(config_indexes, dtype=np.int32)
        config_descriptors = np.stack(config_descriptors, axis=0)
        normalized_runtimes = np.array(normalized_runtimes)

        if set_name == 'train':
            train_len = len(tile_ids)
            print(f'permutating {train_len} training samples')

            new_order = np.random.permutation(np.arange(train_len))

            tile_ids = [tile_ids[i] for i in new_order]
            config_indexes = config_indexes[new_order]
            config_descriptors = config_descriptors[new_order]
            normalized_runtimes = normalized_runtimes[new_order]

            print('creating training tf.data.Dataset')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (tile_ids, config_indexes, config_descriptors, normalized_runtimes))
            tf_dataset = tf_dataset.shuffle(buffer_size=1_000).batch(self.batch_size)

        elif set_name == 'valid':
            print('creating validation tf.data.Dataset')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (tile_ids, config_indexes, config_descriptors, normalized_runtimes))
            tf_dataset = tf_dataset.batch(self.batch_size)

        else:
            print('creating test tf.data.Dataset')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (tile_ids, config_indexes, config_descriptors))
            tf_dataset = tf_dataset.batch(self.batch_size)

        return tf_dataset


class LayoutDataset:
    def __init__(self, batch_size: int, train_sample_fraction: float, subset: str):
        self.root_dir = 'predict-ai-model-runtime/npz_all/npz/layout/'
        self.batch_size = batch_size
        self.n_config_nodes_upper_limit = 300
        max_configs_per_graph = 250
        self.subset = subset
        with tf.device('/cpu:0'):
            self.train_data = self._load_set(
                'train',
                sample_fraction=train_sample_fraction,
                max_configs_per_graph=max_configs_per_graph)
            self.test_data = self._load_set('test')
            self.valid_data = self._load_set('valid', max_configs_per_graph=max_configs_per_graph)

    def _list_filenames(self, set_name: str) -> List[str]:
        filenames_list = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            if len(filenames) == 0:
                continue
            dirpath_split = dirpath.split('/')
            current_set = dirpath_split[-1]
            if current_set != set_name:
                continue
            if self.subset is not None:
                if dirpath_split[-3] != self.subset:
                    continue
                print(dirpath_split)

            for filename in filenames:
                full_filename = os.path.join(dirpath, filename)
                filenames_list.append(full_filename)

        return filenames_list

    def _get_dataset_from_filenames(self, filenames: List[str], set_name: str, max_configs_per_graph: int) -> tf.data.Dataset:
        layout_ids = []
        config_indexes = []
        config_descriptors = []
        graph_descriptors = []
        n_valid_nodes_list = []
        if set_name != 'test':
            runtimes = []

        for filename in tqdm(filenames, desc=f'loading {set_name}'):
            layout_dict = dict(np.load(filename))

            filename_split = filename.split('/')
            filename = filename_split[-1][:-len('.npz')]
            layout_id = f'layout:{filename_split[-4]}:{filename_split[-3]}:{filename}'

            n_configs = len(layout_dict['config_runtime'])
            config_sample = np.arange(n_configs)
            if max_configs_per_graph is not None:
                if n_configs > max_configs_per_graph:
                    config_sample = np.random.choice(config_sample, max_configs_per_graph)

            if set_name != 'test':
                target = layout_dict['config_runtime'][config_sample]
                normalized_target = target / np.min(target)
                target = np.stack([target, normalized_target], axis=-1)
                target = np.log(target)
                runtimes.append(target)

            adjacency_matrix = self.compute_adjacency_matrix(
                layout_dict['edge_index'], len(layout_dict['node_opcode']))
            graph_descriptor = self.compute_graph_descriptor(layout_dict['node_opcode'])
            node_config_feat, n_valid_nodes = self.compute_node_descriptors(
                layout_dict, config_sample, adjacency_matrix)

            layout_ids.append([layout_id]*len(config_sample))
            config_indexes.append(config_sample)
            config_descriptors.append(node_config_feat)
            n_valid_nodes_list.append(n_valid_nodes)
            graph_descriptors.append(np.tile(graph_descriptor, (len(config_sample), 1)))

        layout_ids = np.concatenate(layout_ids, axis=0)
        config_indexes = np.concatenate(config_indexes, axis=0).astype(np.int32)
        config_descriptors = np.concatenate(config_descriptors, axis=0)
        n_valid_nodes_list = np.concatenate(n_valid_nodes_list, axis=0)
        graph_descriptors = np.concatenate(graph_descriptors, axis=0)

        if set_name != 'test':
            runtimes = np.concatenate(runtimes, axis=0).astype(np.float32)

        # From arrays to datasets
        if set_name == 'train':
            # shuffling
            train_len = len(layout_ids)
            print(f'permutating {train_len} training samples')

            new_order = np.random.permutation(np.arange(train_len))

            tile_ids = [layout_ids[i] for i in new_order]
            config_indexes = config_indexes[new_order]
            config_descriptors = config_descriptors[new_order]
            runtimes = runtimes[new_order]
            n_valid_nodes_list = n_valid_nodes_list[new_order]
            graph_descriptors = graph_descriptors[new_order]

            print('creating training tf.data.Dataset')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (tile_ids, config_indexes, config_descriptors,
                 n_valid_nodes_list, graph_descriptors, runtimes))
            tf_dataset = tf_dataset.shuffle(buffer_size=100).batch(self.batch_size)
            # config_descriptors(batch).shape = (batch_size, n_config_nodes_upper_limit, 18)

        elif set_name == 'valid':
            print('creating validation tf.data.Dataset')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (layout_ids, config_indexes, config_descriptors,
                 n_valid_nodes_list, graph_descriptors, runtimes))
            tf_dataset = tf_dataset.batch(self.batch_size)

        else:
            print('creating test tf.data.Dataset')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (layout_ids, config_indexes, config_descriptors,
                 graph_descriptors, n_valid_nodes_list))
            tf_dataset = tf_dataset.batch(self.batch_size)

        return tf_dataset

    def compute_adjacency_matrix(self, edges: np.array, n_nodes: int) -> np.array:
        adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        for edge in edges:
            edge_to, edge_from = edge
            adjacency_matrix[edge_to, edge_from] += 1
        return adjacency_matrix

    def compute_node_descriptors(
            self,
            layout_dict: dict[str, np.array],
            trials_sample: List[int],
            adjacency_matrix: np.array) -> tuple[np.array, np.array]:
        """For up to n_config_nodes_upper_limit, it computes a feature
        description for each of them. It also returns a mask."""

        node_config_feat = layout_dict['node_config_feat'][trials_sample]
        n_config_nodes = node_config_feat.shape[1]
        n_trials = len(trials_sample)
        n_valid_nodes = min(n_config_nodes, self.n_config_nodes_upper_limit)
        n_valid_nodes = np.array([n_valid_nodes] * n_trials, dtype=np.int32)
        # n_valid_nodes.shape == (n_trials,)

        if n_config_nodes > self.n_config_nodes_upper_limit:
            selected_nodes = np.random.choice(
                np.arange(n_config_nodes),
                self.n_config_nodes_upper_limit)
        else:
            selected_nodes = np.arange(n_config_nodes)

        node_config_feat = node_config_feat[:, selected_nodes, :]
        node_config_feat = node_config_feat

        interesting_node_features = np.concatenate([
            np.arange(21, 27),  # shape dims
            np.arange(31, 37),  # reshape/broadcast dims
            np.arange(134, 140),  # phys layout
            np.arange(95, 99),  # conv dims input
            np.arange(101, 105)  # conv dims kernel
        ], axis=0)
        selected_nodes_global = layout_dict['node_config_ids'][selected_nodes]
        node_features = layout_dict['node_feat']
        parent_output_shapes = self.compute_parent_output_shapes(
            adjacency_matrix, selected_nodes_global, node_features)

        node_types = layout_dict['node_opcode'][selected_nodes_global]
        node_types = tf.expand_dims(node_types, axis=1)
        
        node_features_array = node_features[selected_nodes_global]
        node_features_array = node_features_array[:, interesting_node_features]
        node_features_array = np.concatenate(
            [node_features_array, parent_output_shapes, node_types], axis=1)
        node_features_array = magic_log_v(node_features_array)
        node_features_array = np.tile(node_features_array, (n_trials, 1, 1))

        node_descriptors = np.concatenate(
            [node_config_feat, node_features_array], axis=-1)

        padding_size = max(self.n_config_nodes_upper_limit - n_config_nodes, 0)
        if padding_size > 0:
            node_descriptors = np.concatenate(
                [node_descriptors,
                 np.zeros((n_trials, padding_size, node_descriptors.shape[2]))],
                axis=1)

        return node_descriptors, n_valid_nodes

    def compute_parent_output_shapes(
            self,
            adjacency_matrix: np.array,
            node_indexes: List[int],
            node_features: np.array) -> np.array:

        parent_output_shapes_list = []
        for node_index in node_indexes:
            parent_indexes = adjacency_matrix[node_index, :].nonzero()[0]
            parent_shapes = np.zeros(12, dtype=int)

            # check no more than 2 parents
            for i in range(min(len(parent_indexes), 2)):
                parent_index = parent_indexes[i]
                parent_shapes[i*6:(i+1)*6] = node_features[parent_index, np.arange(21, 27)]

            parent_output_shapes_list.append(parent_shapes)

        parent_output_shapes = np.stack(parent_output_shapes_list, axis=0)
        return parent_output_shapes

    def _load_set(self, set_name: str, sample_fraction: float = 1.0, max_configs_per_graph: int = None) -> tf.data.Dataset:
        filenames_list = self._list_filenames(set_name)
        if sample_fraction != 1.0:
            n = len(filenames_list)
            sample_size = int(np.ceil(n * sample_fraction))
            random_sample = np.random.choice(np.arange(n, dtype=np.int64), size=sample_size)
            filenames_list = [filenames_list[i] for i in random_sample]
        dataset = self._get_dataset_from_filenames(filenames_list, set_name, max_configs_per_graph)
        return dataset

    def compute_graph_descriptor(self, node_opcodes):
        nodes, counts = np.unique(node_opcodes, return_counts=True)
        descriptor = np.zeros(120, dtype=np.float32)
        for n, c in zip(nodes, counts):
            # n goes from 1 to 120, so we save it in n-1
            descriptor[n-1] = c
        n_nodes = np.sum(descriptor)
        descriptor = descriptor / n_nodes
        descriptor = np.concatenate([descriptor, np.array([np.log(n_nodes)])])
        return descriptor


if __name__ == '__main__':
    dataset = LayoutDataset(batch_size=64, train_sample_fraction=0.1, subset='nlp')
    # dataset = TileDataset(batch_size=64)

    for i, sample in enumerate(dataset.valid_data):
        print(sample)
        if i == 2:
            break
