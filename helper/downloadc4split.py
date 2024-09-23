from datasets import Dataset, DatasetDict
from datasets import load_dataset

iterable_ds = load_dataset("allenai/c4", "en", split="validation", streaming=True).take(2200)
test = Dataset.from_generator(lambda: (yield from iterable_ds), features=iterable_ds.features)

test.save_to_disk('/pscratch/sd/p/Dir/c4-sample')
