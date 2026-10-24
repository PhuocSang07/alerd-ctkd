from torch.nn.functional import pad
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import json
from transformers import PreTrainedTokenizer

from dataclasses import dataclass


def longest_common_subsequence(a, b, s_i=0, s_j=0, max_spans=None) -> list:
    """Match token end offsets of two tokenizations; `a` and `b` are (len, 2) arrays holding real tokens only.

    max_spans caps the number of spans by even subsampling; None keeps every span.
    """
    m, n = len(a), len(b)
    
    i = s_i
    j = s_j
    result = []

    while i < m and j < n:
        if a[i][1] == 0:
            i += 1
            continue
        if b[j][1] == 0:
            j += 1
            continue
            
        if a[i][1] == b[j][1]:
            result.append((i+1, j+1))
            i += 1
            j += 1
        elif a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

        if i+1 < m and j+1 < n:
            if (a[i][1] == 0 and a[i+1][1] > 0) or (b[j][1] == 0 and b[j+1][1] > 0):
                while j < n and b[j][1] > 0:
                    j += 1

                while i < m and a[i][1] > 0:
                    i += 1

                i += 1
                j += 1
                result.append((i, j))


    size = len(result)
    if max_spans is not None and size > max_spans:
        step = size / max_spans
        return [result[int((i + 1) * step) - 1] for i in range(max_spans)]
    # else:
    #     new_result = [result[i] for i in range(size - 1, -1, -16)]
    #     return new_result[::-1]
            
    return result

def get_pooler_tensor(segments_idxs):
    # Tạo chỉ số segment đã pad cho toàn bộ batch
    padded_idx_batch = []
    max_seg, max_len_all = 0, 0
    pad_multiple = 4

    for seg_idx, max_len in segments_idxs:
        max_len_all = max(max_len_all, max_len)
        max_seg = max(max_seg, len(seg_idx))

        padded = torch.stack([
            pad(x, (0, max_len - len(x)), value=-1)
            for x in seg_idx
        ])  # (num_segments, max_len)

        padded_idx_batch.append(padded)

    # Pad toàn bộ batch về cùng shape (B, max_seg, max_len_all)
    def pad2d(t, h, w):
        return pad(t, (0, w - t.size(1), 0, h - t.size(0)), value=-1)

    # max_seg = int(math.ceil(max_seg / pad_multiple) * pad_multiple)
    padded_idx_batch = torch.stack([
        pad2d(p, max_seg, max_len_all) for p in padded_idx_batch
    ])  # (B, max_seg, max_len_all)

    # Tạo mask và gather từ X
    mask = padded_idx_batch != -1
    safe_idx = padded_idx_batch.masked_fill(~mask, 0)

    return {'safe_idx': safe_idx, 'mask': mask}

def sample_span_bounds(student_offset, teacher_offset, student_start, teacher_start, max_spans=None):
    """Token boundaries (student_idx, teacher_idx) of the aligned spans of one sample.

    The offsets must cover the sample's real tokens only (no padding), so the result does not depend on
    how the sample was batched. The first pair is the response start; span n covers tokens
    [bounds[n][0], bounds[n+1][0]) on the student side and [bounds[n][1], bounds[n+1][1]) on the teacher side.
    """
    student_offset = np.asarray(student_offset)
    teacher_offset = np.asarray(teacher_offset)
    return [(int(student_start), int(teacher_start))] + longest_common_subsequence(
        student_offset, teacher_offset, int(student_start), int(teacher_start), max_spans)


def prepare_pooler_v2(student_starts, student_offset_mapping, student_lengths,
                      teacher_starts, teacher_offset_mapping, teacher_lengths, max_spans=None):
    student_seg_idxs, teacher_seg_idxs = [], []
    for student_start, student_offset, student_len, teacher_start, teacher_offset, teacher_len in zip(
            student_starts, student_offset_mapping, student_lengths,
            teacher_starts, teacher_offset_mapping, teacher_lengths):

        bounds = sample_span_bounds(student_offset[:int(student_len)].numpy(),
                                    teacher_offset[:int(teacher_len)].numpy(),
                                    student_start.item(), teacher_start.item(), max_spans)
        student_seg_idx, teacher_seg_idx = [], []
        student_max_len, teacher_max_len = 1, 1

        for i in range(1, len(bounds)):
            student_seg_idx.append(torch.arange(bounds[i - 1][0], bounds[i][0]))
            teacher_seg_idx.append(torch.arange(bounds[i - 1][1], bounds[i][1]))
            student_max_len = max(student_max_len, student_seg_idx[-1].size(0))
            teacher_max_len = max(teacher_max_len, teacher_seg_idx[-1].size(0))

        student_seg_idxs.append((student_seg_idx, student_max_len))
        teacher_seg_idxs.append((teacher_seg_idx, teacher_max_len))

    return get_pooler_tensor(student_seg_idxs), get_pooler_tensor(teacher_seg_idxs)


class LLMDataset(Dataset):
    def __init__(self, file_path, student_tokenizer, teacher_tokenizer, prompt_max_len=512):

        self.dataset = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.dataset.append(data)

                s_prompt = student_tokenizer(
                    data['prompt'], 
                    max_length=prompt_max_len,
                    truncation=True
                )
                data['prompt'] = student_tokenizer.decode(s_prompt['input_ids'])
                data['s_prompt_len'] = len(s_prompt['input_ids'])

                t_prompt = teacher_tokenizer(data['prompt'])
                data['t_prompt_len'] = len(t_prompt['input_ids'])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (self.dataset[index]['prompt'], self.dataset[index]['output'], 
                self.dataset[index]['s_prompt_len'], self.dataset[index]['t_prompt_len'], index)
    

@dataclass
class LLMDataCollator:
    student_tokenizer: PreTrainedTokenizer = None
    teacher_tokenizer: PreTrainedTokenizer = None
    do_train: bool = True
    max_len: int = 512
    pad_to_multiple_of: int = 4
    return_tensors: str = 'pt'
    padding: bool = True
    return_offsets_mapping: bool = True
    max_spans: int = None

    def __post_init__(self):
        # Span indices start at the prompt length, which only holds with right padding.
        for name, tokenizer in (('student', self.student_tokenizer), ('teacher', self.teacher_tokenizer)):
            if self.do_train and tokenizer is not None and tokenizer.padding_side != 'right':
                raise ValueError(f"{name} tokenizer padding_side is {tokenizer.padding_side!r}; span alignment needs 'right'")


    def __call__(self, batch):
        prompts, fulls, prompt_lengths, teacher_prompt_lengths, example_ids = [], [], [], [], []
        for prompt, output, prompt_length, teacher_prompt_length, example_id in batch:
            example_ids.append(example_id)
            prompts.append(prompt)
            fulls.append(prompt + output)
            prompt_lengths.append(prompt_length)
            teacher_prompt_lengths.append(teacher_prompt_length)

        
        student_inputs = self.student_tokenizer(
            fulls,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping and self.do_train,
        )

        # tokenized_prompts = self.student_tokenizer(
        #     prompts, 
        #     max_length=self.max_len, 
        #     padding=self.padding, 
        #     truncation=True,
        #     return_tensors=self.return_tensors,
        # )
        
        labels = student_inputs["input_ids"].clone().detach()
        input_lengths = student_inputs["attention_mask"].sum(dim=1)
        # labels[labels == self.student_tokenizer.pad_token_id] = -100
        # prompt_lengths = tokenized_prompts["attention_mask"].sum(dim=1)
        prompt_lengths = torch.tensor(prompt_lengths)

        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
            labels[i, (input_lengths[i] + 1):] = -100

        if not self.do_train:
            return student_inputs, None, labels

        student_inputs['example_ids'] = torch.tensor(example_ids)

        student_token_offset_mapping = student_inputs.pop('offset_mapping', None)

        teacher_inputs = self.teacher_tokenizer(
            fulls,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping,
        )
        # teacher_prompts = self.teacher_tokenizer(
        #     prompts, 
        #     max_length=self.max_len, 
        #     padding=self.padding, 
        #     truncation=True,
        #     return_tensors=self.return_tensors,
        # )
        # teacher_prompt_lengths = teacher_prompts["attention_mask"].sum(dim=1)
        teacher_prompt_lengths = torch.tensor(teacher_prompt_lengths)

        teacher_token_offset_mapping = teacher_inputs.pop('offset_mapping', None)

        if student_token_offset_mapping is not None and teacher_token_offset_mapping is not None:
            student_pooler_tensor, teacher_pooler_tensor = prepare_pooler_v2(
                prompt_lengths, student_token_offset_mapping, student_inputs['attention_mask'].sum(dim=1),
                teacher_prompt_lengths, teacher_token_offset_mapping, teacher_inputs['attention_mask'].sum(dim=1),
                self.max_spans)

            student_inputs['pooler_safe_idx'] = student_pooler_tensor['safe_idx']
            student_inputs['pooler_mask'] = student_pooler_tensor['mask']
            teacher_inputs['pooler_safe_idx'] = teacher_pooler_tensor['safe_idx']
            teacher_inputs['pooler_mask'] = teacher_pooler_tensor['mask']

        return student_inputs, teacher_inputs, labels

