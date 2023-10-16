import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import random
from collections import defaultdict
from typing import List, Optional, Tuple
from joblib import Parallel, delayed


def magic_log(x):
    if x <= 0:
        return np.array(-1.0, dtype=np.float32)
    return np.log(x)


magic_log_v = np.vectorize(magic_log)


def compute_adjacency_matrix(edges: np.array, n_nodes: int) -> np.array:
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for edge in edges:
        edge_to, edge_from = edge
        adjacency_matrix[edge_to, edge_from] += 1
    return adjacency_matrix


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
            'layout_tfrecords_v2')
        self.batch_size = batch_size
        self.n_config_nodes_upper_limit = 750
        max_configs_per_graph = 50_000  # None
        self.batch_per_file_size = batch_per_file_size
        self.subset = subset  # {xla, nlp}

        if build_tfrecords:
            self.create_tfrecords(
                'train',
                overwrite=True,
                sample_fraction=train_sample_fraction,
                max_configs_per_graph=max_configs_per_graph)
            self.create_tfrecords('test', overwrite=True)
            self.create_tfrecords('valid', overwrite=True, max_configs_per_graph=1_000)
            exit()

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
            return dataset.prefetch(2)

        dataset = dataset.interleave(
            interleave_fn,
            cycle_length=20,
            num_parallel_calls=16,
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

    def create_tfrecords(
            self,
            set_name: str,
            overwrite: bool,
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
            filenames_list, set_name, max_configs_per_graph, self.tfrecords_dir, overwrite)

    def write_tfrecords(
            self,
            filenames: List[str],
            set_name: str,
            max_trials: int,
            output_folder: str,
            overwrite: bool
    ):

        def write_one_tfrecord(filename, max_trials):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            layout = Layout(filename, max_trials=max_trials)
            layout_id = layout.layout_id

            output_filename = os.path.join(output_folder, layout_id + f':{set_name}.tfrecords')
            if not overwrite and os.path.exists(output_filename):
                return
            
            with tf.io.TFRecordWriter(
                    output_filename,
                    options=tf.io.TFRecordOptions(compression_type='GZIP')) as file_writer:
                layout_feature = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[layout_id.encode('UTF-8')]))
                graph_feature = tf.train.Feature(
                    float_list=tf.train.FloatList(value=layout.global_graph_description))
                for trial_index in range(layout.n_trials):
                    if trial_index % 10_000 == 0:
                        print(layout_id, trial_index)
                        
                    node_config_feat, n_valid_nodes = layout.compute_node_descriptors(
                        trial_index, self.n_config_nodes_upper_limit)
                    node_descriptor_serialized = tf.io.serialize_tensor(node_config_feat).numpy()
                    record_bytes = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'layout_id': layout_feature,
                                'config_index': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[layout.trial_sample[trial_index]])),
                                'node_descriptor': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[node_descriptor_serialized])),
                                'valid_nodes': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[n_valid_nodes])),
                                'graph_descriptor': graph_feature,
                                'target': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=[layout.config_runtime[trial_index]]))
                            }
                        )
                    )
                    file_writer.write(record_bytes.SerializeToString())

        tasks = []
        for filename in filenames:
            tasks.append(delayed(write_one_tfrecord)(filename, max_trials))

        Parallel(n_jobs=3, verbose=11, backend='loky')(tasks)


class Layout:
    def __init__(self, full_filename: str, max_trials: Optional[int]):
        layout_dict = dict(np.load(full_filename))  # type: dict[str, np.ndarray]

        filename_split = full_filename.split('/')
        filename_wo_ext = filename_split[-1][:-len('.npz')]
        self.layout_id = f'layout:{filename_split[-4]}:{filename_split[-3]}:{filename_wo_ext}'

        self.edge_index = layout_dict['edge_index']
        # (n_edges, 2) to <- from

        self.node_feat = layout_dict['node_feat']
        # (n_nodes, 140)

        self.node_opcode = layout_dict['node_opcode']
        # (n_nodes,)

        self.node_config_feat = layout_dict['node_config_feat']
        # (n_trials, n_configurable_nodes, 18)

        self.node_config_ids = layout_dict['node_config_ids']
        # (n_configurable_nodes,)

        self.node_splits = layout_dict['node_splits']
        # ?

        self.config_runtime = layout_dict['config_runtime']
        # (n_trials,)

        n_trials = len(self.config_runtime)

        if max_trials is not None and n_trials > max_trials:
            self.trial_sample = np.random.choice(np.arange(n_trials), max_trials)
            self.node_config_feat = self.node_config_feat[self.trial_sample]
            self.config_runtime = self.config_runtime[self.trial_sample]
            n_trials = max_trials
        else:
            self.trial_sample = np.arange(n_trials)

        self.n_trials = n_trials

        self.config_runtime = np.log(self.config_runtime)
        self.adjacency_matrix = compute_adjacency_matrix(
            self.edge_index, n_nodes=len(self.node_opcode))

        self.global_graph_description = self.compute_graph_description()

        self.parent_shapes = self.compute_parent_output_shapes(
            self.node_config_ids, self.adjacency_matrix, self.node_feat)
        # (n_configurable_nodes, 12)
        # 6 dims per parent, up to 2 parents

    def compute_graph_description(self) -> np.ndarray:
        nodes, counts = np.unique(self.node_opcode, return_counts=True)
        descriptor = np.zeros(120, dtype=np.float32)
        for n, c in zip(nodes, counts):
            # n goes from 1 to 120, so we save it in n-1
            descriptor[n-1] = c
        n_nodes = np.sum(descriptor)
        descriptor = descriptor / n_nodes
        descriptor = np.concatenate([descriptor, np.array([np.log(n_nodes)])])
        return descriptor

    def compute_node_descriptors(
            self,
            trial_index: int,
            max_nodes: Optional[int]) -> Tuple[np.ndarray, int]:

        assert trial_index < self.n_trials
        node_config_feat = self.node_config_feat[trial_index]
        # (n_configurable_nodes, 18)

        n_config_nodes = len(node_config_feat)
        n_valid_nodes = min(n_config_nodes, max_nodes)

        selected_nodes_local = np.arange(n_config_nodes)
        if n_config_nodes > max_nodes:
            selected_nodes_local = np.random.choice(
                selected_nodes_local,
                max_nodes)

        node_config_feat = node_config_feat[selected_nodes_local, :]
        # (n_valid_nodes, 18)

        interesting_node_features = np.concatenate([
            np.arange(21, 27),  # shape dims
            np.arange(31, 37),  # reshape/broadcast dims
            np.arange(95, 99),  # conv dims input
            np.arange(101, 105),  # conv dims kernel
            np.arange(134, 140),  # phys layout
        ], axis=0)

        selected_nodes_global = self.node_config_ids[selected_nodes_local]
        parent_output_shapes = self.parent_shapes[selected_nodes_local]

        node_types = self.node_opcode[selected_nodes_global]
        node_types = tf.expand_dims(node_types, axis=1)

        node_features_array = self.node_feat[:, interesting_node_features]
        node_features_array = node_features_array[selected_nodes_global]
        node_features_array = node_features_array.astype(np.float32)
        parent_output_shapes = parent_output_shapes.astype(np.float32)
        node_config_feat = node_config_feat.astype(np.float32)

        features_that_need_log = np.concatenate(
            [
                node_features_array[:, :-6],
                parent_output_shapes
            ],
            axis=1
        )
        features_that_need_log = magic_log_v(features_that_need_log)
        features = np.concatenate(
            [
                features_that_need_log,
                node_features_array[:, -6:],  # phys layout
                node_config_feat,  # layout configuration
                node_types
            ],
            axis=1
        )

        padding_size = max_nodes - n_config_nodes
        if padding_size > 0:
            features = np.concatenate(
                [
                    features,
                    np.zeros((padding_size, features.shape[1]),
                             dtype=np.float32)],
                axis=0)

        return features, n_valid_nodes

    def compute_parent_output_shapes(
            self,
            configurable_nodes_ids: np.ndarray,
            adjacency_matrix: np.array,
            node_features: np.array) -> np.array:

        parent_output_shapes_list = []
        for node_index in configurable_nodes_ids:
            parent_indexes = adjacency_matrix[node_index, :].nonzero()[0]
            parent_shapes = np.zeros(12, dtype=int)

            # check no more than 2 parents
            for i in range(min(len(parent_indexes), 2)):
                parent_index = parent_indexes[i]
                parent_shapes[i*6:(i+1)*6] = node_features[parent_index, np.arange(21, 27)]

            parent_output_shapes_list.append(parent_shapes)

        parent_output_shapes = np.stack(parent_output_shapes_list, axis=0)
        return parent_output_shapes


if __name__ == '__main__':
    dataset = LayoutDataset(
        batch_size=64,
        train_sample_fraction=1.0,
        subset=None,
        build_tfrecords=True,
        batch_per_file_size=8
    )

    exit()
    dataset = TileDataset(
        batch_size=64,
        batch_per_file_size=8,
        build_tfrecords=False)

    for i, sample in enumerate(dataset.train_data):
        print(np.unique(sample['tile_id'].numpy()))
        if i == 10:
            print(sample)
            break
