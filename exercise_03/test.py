import numpy as np

from exercise_code.data import DataLoader, DummyDataset
from exercise_code.tests import (
    test_dataloader,
    test_dataloader_len,
    test_dataloader_iter,
    save_pickle,
    load_pickle
)





from exercise_code.data.base_dataset import DummyDataset

dataset = DummyDataset(
    root=None,
    divisor=2,
    limit=100
)
# print(
#     "Dataset Length:\t", len(dataset),
#     "\nFirst Element:\t", dataset[0],
#     "\nLast Element:\t", dataset[-1],
# )


batch_size = 3
def build_batches(dataset, batch_size):
    batches = []  # list of all mini-batches
    batch = []  # current mini-batch
    for i in range(len(dataset)):
        batch.append(dataset[i])
        if len(batch) == batch_size:  # if the current mini-batch is full,
            batches.append(batch)  # add it to the list of mini-batches,
            batch = []  # and start a new mini-batch
    return batches

batches = build_batches(
    dataset=dataset,
    batch_size=batch_size
)

def print_batches(batches):
    for i, batch in enumerate(batches):
        print("mini-batch %d:" % i, str(batch))

# print_batches(batches)

def combine_batch_dicts(batch):
    batch_dict = {}
    for data_dict in batch:
        for key, value in data_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    return batch_dict

combined_batches = [combine_batch_dicts(batch) for batch in batches]
# print_batches(combined_batches)

def batch_to_numpy(batch):
    numpy_batch = {}
    for key, value in batch.items():
        numpy_batch[key] = np.array(value)
    return numpy_batch

numpy_batches = [batch_to_numpy(batch) for batch in combined_batches]
# print_batches(numpy_batches)


def build_batch_iterator(dataset, batch_size, shuffle, drop_last):
    # if shuffle:
    #     index_iterator = iter(np.random.permutation(len(dataset)))  # define indices as iterator
    # else:
    #     index_iterator = iter(range(len(dataset)))  # define indices as iterator
    #
    # batch = []
    # for index in index_iterator:  # iterate over indices using the iterator
    #     batch.append(dataset[index])
    #     if len(batch) == batch_size:
    #         yield batch  # use yield keyword to define a iterable generator
    #         batch = []
    if shuffle:
        index_iterator = iter(np.random.permutation(len(self.dataset)))
    else:
        index_iterator = iter(range(len(dataset)))

    batch = []
    n = len(dataset)
    for index in index_iterator:
        batch.append(dataset[index])
        n = n - 1
        if (n == 0 and not drop_last) or (len(batch) == batch_size):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)

            numpy_batch = {}
            for key, value in batch_dict.items():
                numpy_batch[key] = np.array(value)
            yield numpy_batch

            batch = []


batch_iterator = build_batch_iterator(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)
batches = []
for batch in batch_iterator:
    batches.append(batch)

print_batches(
    [batch for batch in batches]
)


from exercise_code.data.dataloader import DataLoader

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

# _ = test_dataloader_len(
#     dataloader=dataloader
# )

from exercise_code.data.dataloader import DataLoader

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)

# _ = test_dataloader_iter(
#     dataloader=dataloader
# )
