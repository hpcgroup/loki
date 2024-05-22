from datasets import Dataset, DatasetDict
from datasets import load_dataset

iterable_ds = load_dataset("bookcorpus", split="train", streaming=True).take(20000)
test = Dataset.from_generator(lambda: (yield from iterable_ds), features=iterable_ds.features)

test.save_to_disk('/pscratch/sd/p/Dir/bookcorpus-sample')
