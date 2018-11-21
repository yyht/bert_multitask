import io
from bert_tokenizer import FullTokenizer
import os
import random


class FileReader(object):
  def __init__(self, filename, sample_rate):
    self.fp = io.open(filename, "rb")
    self.rewind()
    self.sample_rate = sample_rate
    self.sample_original = sample_rate

  def reset(self):
    self.rewind()
    self.sample_rate = self.sample_original

  def rewind(self):
    self.fp.seek(0, os.SEEK_SET)
    # skip header
    self.fp.readline()

  def __len__(self):
    count = 0
    self.rewind()
    for _ in self.fp:
      count += 1
    self.rewind()
    return int(count * self.sample_original)

  def valid(self):
    return self.sample_rate > 0

  def readline(self):
    if self.sample_rate <= 0:
      return
    fp = self.fp
    line = fp.readline()

    # reach EOF
    if len(line) == 0:
      self.sample_rate -= 1
      if self.sample_rate > 0:
        self.rewind()
        line = fp.readline()
      else:
        return
    if self.sample_rate >= 1:
      return line
    if random.uniform(0, 1) < self.sample_rate:
      return line


class InputPipeline(object):
  def __init__(self, vocab_file, input_files, max_seq_len, sample_rate):
    self.input_files = input_files
    self.tokenizer = FullTokenizer(vocab_file=vocab_file)
    self.max_seq_len = max_seq_len
    self.input_files = input_files
    # self.readers = [(i, FileReader(input_file, rate)) for i, (input_file, rate)
    #                in enumerate(zip(input_files, sample_rate))]

    self.sample_rate = sample_rate
    self.task_id = -1

  def set_reader(self, filename):
    if filename == "train.tsv":
      sample_rate = self.sample_rate
    else:
      sample_rate = [1] * len(self.sample_rate)
    self.readers = [(i, FileReader(os.path.join(input_file, filename), rate)) for i, (input_file, rate)
                    in enumerate(zip(self.input_files, sample_rate))]
    if self.task_id !=-1:
      self.readers = [self.readers[self.task_id]]


  def num_categories(self):
    return 2

  def num_train_examples(self):
    self.set_reader("train.tsv")
    value = sum([len(reader) for _, reader in self.readers])
    return value

  def num_eval_examples(self):
    self.set_reader("dev.tsv")
    value = sum([len(reader) for _, reader in self.readers])
    return value

  def num_test_examples(self):
    self.set_reader("test.tsv")
    value = sum([len(reader) for _, reader in self.readers])
    return value

  def set_task_id(self,task_id):
    self.task_id = task_id

  def iter_train(self):
    return self.iter_data("train.tsv")

  def iter_test(self):
    return self.iter_data("test.tsv")

  def iter_eval(self):
    return self.iter_data("dev.tsv")

  def iter_data(self, filename):
    self.set_reader(filename)
    index = 0
    while len(self.readers):
      for i, reader in self.readers:
        line = reader.readline()
        if line is None:
          continue
        line = line.strip()
        if len(line) == 0:
          continue
        label, _, _, text_a, text_b = line.split(b'\t')
        input_ids, input_mask, segment_ids = self.tokenizer.convert_pairs(text_a, text_b, self.max_seq_len)
        yield {
          "input_ids": input_ids,
          "input_mask": input_mask,
          "segment_ids": segment_ids,
          "loss_mask": i,
          "label_ids": int(label)
        }
        index += 1
      pop = -1
      for i, (_, reader) in enumerate(self.readers):
        if not reader.valid():
          pop = i
          break
      if pop != -1:
        self.readers.pop(pop)


if __name__ == "__main__":
  reader = FileReader("data/dev.tsv", 5)

  print(len(reader))

  none_count = 0
  b_count = 0
  while reader.valid():
    res = reader.readline()
    if res is None:
      none_count += 1
    else:
      b_count += 1
  print(none_count, b_count)

  pipeline = InputPipeline("chinese_L-12_H-768_A-12/vocab.txt",
                           ["data", "data"],
                           30, [1, 1])
  count = 0
  pipeline.set_task_id(1)
  for line in pipeline.iter_eval():
    #print(line)
    count +=1
  print(count)

  count = 0
  for line in pipeline.iter_train():
    count +=1
  print(count)
