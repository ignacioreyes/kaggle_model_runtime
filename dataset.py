import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import random
from collections import defaultdict
from typing import List


def magic_log(x):
    if x <= 0:
        return np.array(-1.0, dtype=np.float32)
    return np.log(x)


magic_log_v = np.vectorize(magic_log)


class TileDataset:
    def __init__(
            self,
            batch_size: int,
            batch_per_file_size: int,
            build_tfrecords: bool):

        self.root_dir = 'predict-ai-model-runtime/npz_all/npz/tile/xla'
        self.tfrecords_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'tile_tfrecords')
        self.batch_size = batch_size
        self.batch_per_file_size = batch_per_file_size

        if build_tfrecords:
            self.create_tfrecords('train')
            self.create_tfrecords('test')
            self.create_tfrecords('valid')

        with tf.device('/cpu:0'):
            self.train_data = self.load_tfrecords('train')
            self.valid_data = self.load_tfrecords('valid')
            self.test_data = self.load_tfrecords('test')

    def create_tfrecords(self, set_name: str) -> None:
        filenames_list = self._list_filenames(set_name)
        if not os.path.exists(self.tfrecords_dir):
            os.mkdir(self.tfrecords_dir)
        self.write_tfrecords(
            filenames_list, set_name, self.tfrecords_dir)

    def write_tfrecords(
            self,
            filenames: List[str],
            set_name: str,
            output_folder: str):

        for filename in tqdm(filenames, desc=f'loading {set_name}'):
            tile_dict = dict(np.load(filename))

            filename_split = filename.split('/')
            filename = filename_split[-1][:-len('.npz')]
            tile_id = f'tile:xla:{filename}'

            output_filename = os.path.join(
                output_folder,
                tile_id + f':{set_name}.tfrecords')

            n_configs = len(tile_dict['config_feat'])
            target = -100.0 * np.ones(n_configs)
            if set_name != 'test':
                target = tile_dict['config_runtime'] / tile_dict['config_runtime_normalizers']
                target = np.log(target)
            # target.shape == (n_configs,)

            config_descriptor = tile_dict['config_feat']
            # config_descriptor.shape == (n_configs, 24)

            new_order = np.random.permutation(np.arange(n_configs))
            target = target[new_order]
            config_descriptor = config_descriptor[new_order]

            with tf.io.TFRecordWriter(
                    output_filename,
                    options=tf.io.TFRecordOptions(compression_type='GZIP')) as file_writer:

                layout_feature = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tile_id.encode('UTF-8')]))

                for i_sample in range(n_configs):
                    config_descriptor_serialized = tf.io.serialize_tensor(
                        config_descriptor[i_sample]).numpy()

                    record_bytes = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'tile_id': layout_feature,
                                'config_index': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[new_order[i_sample]])),
                                'config_descriptor': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[config_descriptor_serialized])),
                                'target': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=[target[i_sample]]))
                            }
                        )
                    )
                    file_writer.write(record_bytes.SerializeToString())

    @staticmethod
    def tfrecord_decoder(record_bytes):
        parsed_example = tf.io.parse_single_example(
            record_bytes,

            # schema
            {'tile_id': tf.io.FixedLenFeature([], dtype=tf.string),
             'config_index': tf.io.FixedLenFeature([], dtype=tf.int64),
             'config_descriptor': tf.io.FixedLenFeature([], dtype=tf.string),
             'target': tf.io.FixedLenFeature([], dtype=tf.float32)
             }
        )
        parsed_example['config_descriptor'] = tf.io.parse_tensor(
            parsed_example['config_descriptor'], tf.float32
        )
        return parsed_example

    def _list_filenames(self, set_name: str) -> List[str]:
        filenames_list = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            if len(filenames) == 0:
                continue
            dirpath_split = dirpath.split('/')
            current_set = dirpath_split[-1]
            if current_set != set_name:
                continue

            for filename in filenames:
                full_filename = os.path.join(dirpath, filename)
                filenames_list.append(full_filename)

        return filenames_list

    def load_tfrecords(self, set_name: str) -> tf.data.Dataset:
        assert set_name in ('train', 'valid', 'test')

        tfrecords_file_list = os.listdir(self.tfrecords_dir)
        filenames = []

        for filename in tfrecords_file_list:
            f = filename[:-(len('.tfrecords'))].split(':')
            if f[-1] != set_name:
                continue

            filenames.append(
                os.path.join(self.tfrecords_dir, filename))

        random.shuffle(filenames)  # inplace
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        def interleave_fn(filename: str) -> tf.data.Dataset:
            dataset = tf.data.TFRecordDataset(
                filename, compression_type='GZIP')
            dataset = dataset.map(
                self.tfrecord_decoder, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=20)
            if set_name == 'train':
                dataset = dataset.batch(
                    self.batch_per_file_size, drop_remainder=True)
            return dataset

        dataset = dataset.interleave(
            interleave_fn,
            cycle_length=20,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

        if set_name == 'train':
            batch_size = (self.batch_size // self.batch_per_file_size) * self.batch_per_file_size
            batch_size = int(batch_size)
            dataset = dataset.rebatch(batch_size)
        else:
            dataset = dataset.batch(self.batch_size)

        final_dataset = dataset.prefetch(10)

        return final_dataset


class LayoutDataset:
    def __init__(
            self,
            batch_size: int,
            train_sample_fraction: float,
            subset: [str, None],
            build_tfrecords: bool,
            batch_per_file_size: int
    ):
        self.root_dir = 'predict-ai-model-runtime/npz_all/npz/layout/'
        self.tfrecords_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'layout_tfrecords')
        self.batch_size = batch_size
        self.n_config_nodes_upper_limit = 500
        max_configs_per_graph = 10_000  # None
        self.batch_per_file_size = batch_per_file_size
        self.subset = subset  # {xla, nlp}

        if build_tfrecords:
            self.create_tfrecords(
                'train',
                sample_fraction=train_sample_fraction,
                max_configs_per_graph=max_configs_per_graph)
            self.create_tfrecords('test')
            self.create_tfrecords('valid', max_configs_per_graph=max_configs_per_graph)

        with tf.device('/cpu:0'):
            self.train_data = self.load_tfrecords('train')
            self.test_data = self.load_tfrecords('test')
            self.valid_data = self.load_tfrecords('valid')

    @staticmethod
    def tfrecord_decoder(record_bytes):
        parsed_example = tf.io.parse_single_example(
            record_bytes,

            # schema
            {'layout_id': tf.io.FixedLenFeature([], dtype=tf.string),
             'config_index': tf.io.FixedLenFeature([], dtype=tf.int64),
             'node_descriptor': tf.io.FixedLenFeature([], dtype=tf.string),
             'valid_nodes': tf.io.FixedLenFeature([], dtype=tf.int64),
             'graph_descriptor': tf.io.FixedLenFeature([121], dtype=tf.float32),
             'target': tf.io.FixedLenFeature([], dtype=tf.float32)
             }
        )
        parsed_example['node_descriptor'] = tf.io.parse_tensor(
            parsed_example['node_descriptor'], tf.float32
        )
        return parsed_example

    def load_tfrecords(self, set_name: str) -> tf.data.Dataset:
        assert set_name in ('train', 'valid', 'test')

        tfrecords_file_list = os.listdir(self.tfrecords_dir)
        filenames_dict = defaultdict(list)

        for filename in tfrecords_file_list:
            f = filename[:-(len('.tfrecords'))].split(':')
            if f[-1] != set_name:
                continue

            if set_name == 'train':
                key = ':'.join(f[:3])
            else:
                key = 'all_filenames'
            filenames_dict[key].append(
                os.path.join(self.tfrecords_dir, filename))

        datasets = []
        for v in filenames_dict.values():
            random.shuffle(v)  # inplace
            datasets.append(self.build_dataset_from_filenames(v, set_name))

        if set_name == 'train':
            datasets = [dataset.repeat() for dataset in datasets]
            final_dataset = tf.data.Dataset.sample_from_datasets(
                datasets,
                stop_on_empty_dataset=True
            )
            batch_size = (self.batch_size // self.batch_per_file_size) * self.batch_per_file_size
            batch_size = int(batch_size)
            final_dataset = final_dataset.rebatch(batch_size)
        else:
            final_dataset = datasets[0]
            final_dataset = final_dataset.batch(self.batch_size)

        final_dataset = final_dataset.prefetch(10)

        return final_dataset

    def build_dataset_from_filenames(self, filenames: List[str], set_name: str) -> tf.data.Dataset:
        assert set_name in ('train', 'valid', 'test')
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        def interleave_fn(filename: str) -> tf.data.Dataset:
            dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')
            dataset = dataset.map(self.tfrecord_decoder, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=20)
            if set_name == 'train':
                dataset = dataset.batch(self.batch_per_file_size, drop_remainder=True)
            return dataset

        dataset = dataset.interleave(
            interleave_fn,
            cycle_length=10, num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)
        return dataset

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

        node_features_array = node_features_array.astype(np.float32)
        parent_output_shapes = parent_output_shapes.astype(np.float32)
        node_config_feat = node_config_feat.astype(np.float32)
        
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
                 np.zeros(
                     (n_trials, padding_size, node_descriptors.shape[2]),
                     dtype=np.float32)],
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

    def create_tfrecords(
            self,
            set_name: str,
            sample_fraction: float = 1.0,
            max_configs_per_graph: int = None):
        
        filenames_list = self._list_filenames(set_name)
        if sample_fraction != 1.0:
            n = len(filenames_list)
            sample_size = int(np.ceil(n * sample_fraction))
            random_sample = np.random.choice(np.arange(n, dtype=np.int64), size=sample_size)
            filenames_list = [filenames_list[i] for i in random_sample]

        if not os.path.exists(self.tfrecords_dir):
            os.mkdir(self.tfrecords_dir)
        self.write_tfrecords(
            filenames_list, set_name, max_configs_per_graph, self.tfrecords_dir)

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

    def write_tfrecords(
            self,
            filenames: List[str],
            set_name: str,
            max_configs_per_graph: int,
            output_folder: str):
        
        for filename in filenames:  # tqdm(filenames, desc=f'loading {set_name}'):
            layout_dict = dict(np.load(filename))

            filename_split = filename.split('/')
            filename = filename_split[-1][:-len('.npz')]
            layout_id = f'layout:{filename_split[-4]}:{filename_split[-3]}:{filename}'
            output_filename = os.path.join(output_folder, layout_id+f':{set_name}.tfrecords')

            n_configs = len(layout_dict['config_runtime'])
            config_sample = np.arange(n_configs)
            if max_configs_per_graph is not None:
                if n_configs > max_configs_per_graph:
                    config_sample = np.random.choice(config_sample, max_configs_per_graph)
                    n_configs = max_configs_per_graph

            target = -100.0*np.ones(n_configs)
            if set_name != 'test':
                target = layout_dict['config_runtime'][config_sample]
                # normalized_target = target / np.min(target)
                # target = np.stack([target, normalized_target], axis=-1)
                target = np.log(target)

            adjacency_matrix = self.compute_adjacency_matrix(
                layout_dict['edge_index'], len(layout_dict['node_opcode']))
            graph_descriptor = self.compute_graph_descriptor(layout_dict['node_opcode'])
            node_config_feat, n_valid_nodes = self.compute_node_descriptors(
                layout_dict, config_sample, adjacency_matrix)

            with tf.io.TFRecordWriter(
                    output_filename,
                    options=tf.io.TFRecordOptions(compression_type='GZIP')) as file_writer:
                layout_feature = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[layout_id.encode('UTF-8')]))
                graph_feature = tf.train.Feature(
                    float_list=tf.train.FloatList(value=graph_descriptor))
                for i_sample in tqdm(range(n_configs)):
                    node_descriptor_serialized = tf.io.serialize_tensor(node_config_feat[i_sample]).numpy()
                    record_bytes = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'layout_id': layout_feature,
                                'config_index': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[config_sample[i_sample]])),
                                'node_descriptor': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[node_descriptor_serialized])),
                                'valid_nodes': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[n_valid_nodes[i_sample]])),
                                'graph_descriptor': graph_feature,
                                'target': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=[target[i_sample]]))
                            }
                        )
                    )
                    file_writer.write(record_bytes.SerializeToString())


if __name__ == '__main__':
    # dataset = LayoutDataset(
    #     batch_size=64,
    #     train_sample_fraction=1.0,
    #     subset=None,
    #     build_tfrecords=False)

    dataset = TileDataset(
        batch_size=64,
        batch_per_file_size=8,
        build_tfrecords=False)

    for i, sample in enumerate(dataset.train_data):
        print(np.unique(sample['tile_id'].numpy()))
        if i == 10:
            print(sample)
            break
