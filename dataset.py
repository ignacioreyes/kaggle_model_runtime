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
            'tile_tfrecords_v2')
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

            graph_descriptor = self.compute_graph_description(
                tile_dict['node_opcode'])

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
                graph_feature = tf.train.Feature(
                    float_list=tf.train.FloatList(value=graph_descriptor))

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
                                'graph_descriptor': graph_feature,
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
             'graph_descriptor': tf.io.FixedLenFeature([121], dtype=tf.float32),
             'target': tf.io.FixedLenFeature([], dtype=tf.float32)
             }
        )
        parsed_example['config_descriptor'] = tf.io.parse_tensor(
            parsed_example['config_descriptor'], tf.float32
        )
        return parsed_example

    def compute_graph_description(self, node_opcode) -> np.ndarray:
        nodes, counts = np.unique(node_opcode, return_counts=True)
        descriptor = np.zeros(120, dtype=np.float32)
        for n, c in zip(nodes, counts):
            # n goes from 1 to 120, so we save it in n-1
            descriptor[n-1] = c
        n_nodes = np.sum(descriptor)
        descriptor = descriptor / n_nodes
        descriptor = np.concatenate([descriptor, np.array([np.log(n_nodes)])])
        return descriptor

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
            if set_name == 'train':
                dataset = dataset.shuffle(buffer_size=30)
                dataset = dataset.take(1250)
                dataset = dataset.batch(
                    self.batch_per_file_size, drop_remainder=True)
            return dataset

        dataset = dataset.interleave(
            interleave_fn,
            cycle_length=100,
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
            dataset_take: int,
            subset: [str, None],
            build_tfrecords: bool,
            batch_per_file_size: int
    ):
        self.root_dir = 'predict-ai-model-runtime/npz_all/npz/layout/'
        self.tfrecords_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'layout_tfrecords_v5')
        self.batch_size = batch_size
        n_config_nodes_upper_limit = 1000
        max_trials_training = 7500  # None
        self.n_siblings = 3
        self.batch_per_file_size = batch_per_file_size
        self.dataset_take = dataset_take

        if build_tfrecords:
            self.create_tfrecords(
                'train',
                overwrite=False,
                n_siblings=self.n_siblings,
                max_trials_per_graph=max_trials_training,
                max_nodes=n_config_nodes_upper_limit
            )
            self.create_tfrecords(
                'test',
                overwrite=False,
                n_siblings=self.n_siblings,
                max_nodes=2400
            )
            self.create_tfrecords(
                'valid',
                overwrite=False,
                n_siblings=self.n_siblings,
                max_trials_per_graph=1_000,
                max_nodes=2400
            )

        with tf.device('/cpu:0'):
            self.train_data = self.load_tfrecords('train', subset)
            self.test_data = self.load_tfrecords('test', subset)
            self.valid_data = self.load_tfrecords('valid', subset)

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

    def load_tfrecords(self, set_name: str, subset: Optional[str]) -> tf.data.Dataset:
        assert set_name in ('train', 'valid', 'test')
        if subset is not None:
            return self.load_tfrecords_subset(set_name, subset)

        tfrecords_file_list = os.listdir(self.tfrecords_dir)
        filenames_dict = defaultdict(list)

        for filename in tfrecords_file_list:
            f = filename[:-(len('.tfrecords'))].split(':')
            if f[-1] != set_name:
                continue

            file_subset = ':'.join(f[:3])
            if set_name == 'train':
                key = file_subset
            else:
                key = 'all_filenames'
            filenames_dict[key].append(
                os.path.join(self.tfrecords_dir, filename))

        datasets = []
        for subset, v in filenames_dict.items():
            random.shuffle(v)  # inplace
            if set_name == 'test':
                take = 10000
            elif set_name == 'valid':
                take = 1000
            elif 'xla' in subset:
                take = self.dataset_take * 3
            else:
                take = self.dataset_take

            print(set_name, subset, take)
            datasets.append(self.build_dataset_from_filenames(v, set_name, take))

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

        final_dataset = final_dataset.prefetch(2)

        return final_dataset

    def load_tfrecords_subset(self, set_name: str, subset: str) -> tf.data.Dataset:
        assert subset in (
            'nlp:random',
            'nlp:default',
            'xla:random',
            'xla:default'
        )
        tfrecords_file_list = os.listdir(self.tfrecords_dir)
        filenames = []

        for filename in tfrecords_file_list:
            f = filename[:-(len('.tfrecords'))].split(':')

            file_set_name = f[-1]
            if file_set_name != set_name:
                continue

            file_subset = ':'.join(f[1:3])
            if file_subset != subset:
                continue

            filenames.append(
                os.path.join(self.tfrecords_dir, filename))

        random.shuffle(filenames)
        dataset = self.build_dataset_from_filenames(filenames, set_name)

        if set_name == 'train':
            batch_size = (self.batch_size // self.batch_per_file_size) * self.batch_per_file_size
            batch_size = int(batch_size)
            dataset = dataset.rebatch(batch_size)
        else:
            dataset = dataset.batch(self.batch_size)

        dataset = dataset.prefetch(3)

        return dataset

    def build_dataset_from_filenames(self, filenames: List[str], set_name: str, take: int) -> tf.data.Dataset:
        assert set_name in ('train', 'valid', 'test')
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        def interleave_fn(filename: str) -> tf.data.Dataset:
            dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')
            dataset = dataset.map(self.tfrecord_decoder, num_parallel_calls=4)
            if set_name == 'train':
                # dataset = dataset.shuffle(buffer_size=20)
                dataset = dataset.take(take)
                dataset = dataset.batch(self.batch_per_file_size, drop_remainder=True)
            return dataset

        dataset = dataset.interleave(
            interleave_fn,
            cycle_length=50,
            num_parallel_calls=8,
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

            for filename in filenames:
                full_filename = os.path.join(dirpath, filename)
                filenames_list.append(full_filename)

        return filenames_list

    def create_tfrecords(
            self,
            set_name: str,
            overwrite: bool,
            n_siblings: int,
            max_nodes: int,
            max_trials_per_graph: int = None):
        
        filenames_list = self._list_filenames(set_name)

        if not os.path.exists(self.tfrecords_dir):
            os.mkdir(self.tfrecords_dir)
        self.write_tfrecords(
            filenames_list,
            set_name,
            n_siblings,
            max_trials_per_graph,
            self.tfrecords_dir,
            overwrite,
            max_nodes
        )

    def write_tfrecords(
            self,
            filenames: List[str],
            set_name: str,
            n_siblings: int,
            max_trials: int,
            output_folder: str,
            overwrite: bool,
            max_nodes: int
    ):

        def write_one_tfrecord(filename: str, n_siblings: int, max_trials: int, max_nodes: int):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            layout = Layout(filename, n_siblings=n_siblings,
                            max_trials=max_trials, max_nodes=max_nodes)
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
                    if trial_index % 2_500 == 0:
                        print(layout_id, trial_index)
                        
                    node_config_feat, n_valid_nodes = layout.compute_node_descriptors(trial_index)
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
            # write_one_tfrecord(filename, n_siblings, max_trials)
            tasks.append(delayed(write_one_tfrecord)(filename, n_siblings, max_trials, max_nodes))

        Parallel(n_jobs=6, verbose=11, backend='loky')(tasks)


class Layout:
    def __init__(self, full_filename: str, n_siblings: int,
                 max_nodes: int, max_trials: Optional[int]):
        layout_dict = dict(np.load(full_filename))  # type: dict[str, np.ndarray]
        self.n_siblings = n_siblings

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

        self.remove_duplicated_configs()

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

        self.parent_info = self.compute_parent_info(
            self.node_config_ids, self.adjacency_matrix, self.node_feat)
        # (n_configurable_nodes, 12)
        # 6 dims per parent, up to 2 parents
        self.configurable_siblings = self.get_all_configurable_siblings()
        self.max_nodes = max_nodes
        self.node_probs = self.get_node_probs()

    def remove_duplicated_configs(self):
        unique_configs, inverse_index = np.unique(
            self.node_config_feat, return_inverse=True, axis=0)

        times = []
        for i, config in enumerate(unique_configs):
            config_times = self.config_runtime[inverse_index == i]
            average_time = np.mean(config_times)
            times.append(average_time)

        self.node_config_feat = unique_configs
        self.config_runtime = np.array(times)

    def get_node_probs(self) -> Optional[np.ndarray]:
        if len(self.node_config_ids) <= self.max_nodes:
            return None
        node_stds = np.max(np.std(self.node_config_feat, axis=0), axis=1)
        node_stds = np.clip(node_stds, a_min=1e-6, a_max=None)
        return node_stds / np.sum(node_stds)

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
            trial_index: int) -> Tuple[np.ndarray, int]:
        """
        sibling_info:
            sibling output shape: n_siblings*6
            sibling layout and is_layout_equal: n_sibling*(18+1)
        node features:
            node output shape: 6
            reshape/broadcast dims: 6
            conv dims input: 4
            conv dims kernel: 4
            physical layout: 6
        node layout: 18
        parents:
            parents output shapes: 2*6
            parents phys layout: 2*6
        parents opcodes: 2
        siblings opcodes: n_siblings
        node opcode: 1

        if n_siblings == 3, -> 149 features
        :param trial_index:
        :return:
        """

        assert trial_index < self.n_trials
        node_layout_configuration = self.node_config_feat[trial_index]
        # (n_configurable_nodes, 18)

        n_config_nodes = len(node_layout_configuration)
        n_valid_nodes = min(n_config_nodes, self.max_nodes)

        selected_nodes_local = np.arange(n_config_nodes)
        if n_config_nodes > self.max_nodes:
            selected_nodes_local = np.random.choice(
                selected_nodes_local,
                self.max_nodes,
                p=self.node_probs
            )

        node_layout_configuration = node_layout_configuration[selected_nodes_local, :]
        # (n_valid_nodes, 18)

        interesting_node_features = np.concatenate([
            np.arange(21, 27),  # shape dims
            np.arange(31, 37),  # reshape/broadcast dims
            np.arange(95, 99),  # conv dims input
            np.arange(101, 105),  # conv dims kernel
            np.arange(134, 140),  # phys layout
        ], axis=0)

        selected_nodes_global = self.node_config_ids[selected_nodes_local]

        parent_output_and_layout = self.parent_info[selected_nodes_local, :-2]
        parent_opcodes = self.parent_info[selected_nodes_local, -2:]

        sibling_info = self.compute_sibling_info(
            trial_index, selected_nodes_local, node_layout_configuration)

        node_types = self.node_opcode[selected_nodes_global]
        node_types = tf.expand_dims(node_types, axis=1)

        node_features_array = self.node_feat[:, interesting_node_features]
        node_features_array = node_features_array[selected_nodes_global]

        features = np.concatenate(
            [
                sibling_info[:, :-self.n_siblings],
                node_features_array,
                node_layout_configuration,  # layout configuration
                parent_output_and_layout,
                # all the following are opcodes
                parent_opcodes,
                sibling_info[:, -self.n_siblings:],
                node_types
            ],
            axis=1
        )

        padding_size = self.max_nodes - n_config_nodes
        if padding_size > 0:
            features = np.concatenate(
                [
                    features,
                    np.zeros((padding_size, features.shape[1]),
                             dtype=np.float32)],
                axis=0)

        features = features.astype(np.float32)
        return features, n_valid_nodes

    def compute_parent_info(
            self,
            configurable_nodes_ids: np.ndarray,
            adjacency_matrix: np.array,
            node_features: np.array) -> np.array:

        parents_info_list = []
        for node_index in configurable_nodes_ids:
            parent_indexes = adjacency_matrix[node_index, :].nonzero()[0]

            # check no more than 2 parents
            parent_indexes = parent_indexes[:2]
            # first parent has to be the largest
            if len(parent_indexes) == 2:
                p0_size = node_features[parent_indexes[0], 28]
                p1_size = node_features[parent_indexes[1], 28]
                if p0_size < p1_size:
                    parent_indexes[0], parent_indexes[1] = parent_indexes[1], parent_indexes[0]

            # parent info:
            # from 0 to 5: shapes 1st parent
            # 6 to 11: shapes 2nd parent
            # 12 to 17: phys layout 1st parent
            # 18 to 23: phys layout 2nd parent
            # 24, 25: parent opcodes
            parent_info = np.zeros(26, dtype=int)

            for i in range(len(parent_indexes)):
                parent_index = parent_indexes[i]
                parent_info[i*6:i*6+6] = node_features[parent_index, np.arange(21, 27)]
                parent_info[i*6+12:i*6+18] = node_features[parent_index, np.arange(134, 140)]
                parent_info[24+i] = self.node_opcode[parent_index]

            parents_info_list.append(parent_info)

        parents_info = np.stack(parents_info_list, axis=0)
        return parents_info

    def get_all_configurable_siblings(self) -> List[List[int]]:
        configurable_siblings = []
        for local_node_index in range(len(self.node_config_ids)):
            node_index = self.node_config_ids[local_node_index]
            parent_indexes = self.adjacency_matrix[node_index, :].nonzero()[0]

            # siblings
            siblings = set()
            for i in range(len(parent_indexes)):
                parent_id = parent_indexes[i]
                siblings_indexes = self.adjacency_matrix[:, parent_id].nonzero()[0]
                siblings.update(siblings_indexes)

            siblings.remove(node_index)
            siblings = list(siblings)
            # only configurable, sorry :/
            siblings = [s for s in siblings if s in self.node_config_ids]
            configurable_siblings.append(siblings)
        return configurable_siblings

    def compute_sibling_info(
            self,
            trial_index: int,
            selected_nodes_local: np.ndarray,
            node_layouts: np.ndarray):

        sibling_info_list = []
        for i_node, local_node_index in enumerate(selected_nodes_local):
            siblings = self.configurable_siblings[local_node_index]
            random.shuffle(siblings)

            # 6 output shape
            # 18 + 1 layout
            # 1 opcode
            sibling_output_shape = np.zeros(self.n_siblings*6, dtype=int)
            layout_and_opcode = np.zeros(self.n_siblings * (18+1+1), dtype=int)
            node_layout = node_layouts[i_node]
            for i_sibling, sibling_global_index in enumerate(siblings[:self.n_siblings]):
                sibling_output_shape[i_sibling * 6:i_sibling * 6 + 6] = self.node_feat[sibling_global_index, np.arange(21, 27)]

                local_sibling_index = np.where(self.node_config_ids == sibling_global_index)[0][0]
                sibling_layout = self.node_config_feat[trial_index, local_sibling_index, :]
                layout_and_opcode[i_sibling*19:i_sibling*19+18] = sibling_layout
                is_layout_equal = np.all(node_layout == sibling_layout).astype(int)*2-1
                layout_and_opcode[i_sibling * 19 + 18] = is_layout_equal
                layout_and_opcode[-(i_sibling+1)] = self.node_opcode[sibling_global_index]

            sibling_info = np.concatenate([sibling_output_shape, layout_and_opcode])
            sibling_info_list.append(sibling_info)

        sibling_info_array = np.stack(sibling_info_list, axis=0)
        return sibling_info_array


if __name__ == '__main__':
    # dataset = TileDataset(
    #     batch_size=64,
    #     batch_per_file_size=8,
    #     build_tfrecords=False)

    dataset = LayoutDataset(
        batch_size=128,
        dataset_take=1500,
        subset=None,
        build_tfrecords=True,
        batch_per_file_size=8
    )

    for batch in dataset.train_data:
        for k, v in batch.items():
            print(k, v.shape)
        break

    for batch in dataset.test_data:
        for k, v in batch.items():
            print(k, v.shape)
        break
