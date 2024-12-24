# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from lmdeploy.utils import get_logger
from torch.nn.utils.rnn import pad_sequence

logger = get_logger('lmdeploy')

InputIdsType = List[int]
InputEmbsType = Union[None, List[Union[torch.Tensor, np.ndarray]]]
InputEmbRngsType = Union[None, List[Tuple[int, int]]]
PromptType = Union[str, List[Dict]]


def _get_event_loop():
    """get event loop."""
    try:
        event_loop = asyncio.get_event_loop()
    except Exception:
        logger.warning('Can not found event loop in current thread.'
                       ' Create a new event loop.')
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


class LogitsMixin:
    """Helper class to calculate logits and ppl."""

    def prepare_inputs(self, prompts: Union[PromptType, List[PromptType]]):
        if hasattr(self, '_convert_prompts'):
            prompts = self._convert_prompts(prompts)
        need_list_wrap = isinstance(prompts, str) or isinstance(
            prompts[0], Dict)
        prompts = [prompts] if need_list_wrap else prompts

        decorated = []
        input_ids = []
        input_embeddings = []
        input_embedding_ranges = []
        for prompt in prompts:
            out = _get_event_loop().run_until_complete(
                self._get_prompt_input(prompt,
                                       do_preprocess=True,
                                       sequence_start=True,
                                       adapter_name=None))
            decorated.append(out['prompt'])
            input_ids.append(out['input_ids'])
            input_embeddings.append(out.get('input_embeddings', None))
            input_embedding_ranges.append(
                out.get('input_embedding_ranges', None))

        outputs = dict(prompts=decorated, input_ids=input_ids)
        if not any(input_embeddings):
            input_embeddings = None
            input_embedding_ranges = None
        outputs['input_embeddings'] = input_embeddings
        outputs['input_embedding_ranges'] = input_embedding_ranges

        return outputs

    def get_logits(
        self,
        input_ids: Union[InputIdsType, List[InputIdsType]],
        input_embeddings: Union[InputEmbsType, List[InputEmbsType]] = None,
        input_embedding_ranges: Union[InputEmbRngsType,
                                      List[InputEmbRngsType]] = None):
        """Get logits given a list of input tokens.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids
        """
        assert len(input_ids) > 0
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        for input_id in input_ids:
            assert len(input_id) > 0

        bs = len(input_ids)
        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.hf_tm_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (bs * vocab_size * 4)

        n_max_iter = np.ceil(
            max([len(input_id)
                 for input_id in input_ids]) / max_input_len).astype(int)

        index_range_starts = []
        index_range_ends = []
        for input_id in input_ids:
            index_range_start = np.array(
                [i * max_input_len for i in range(n_max_iter)])
            index_range_end = index_range_start + max_input_len
            index_range_start[index_range_start >= len(input_id)] = len(
                input_id)
            index_range_end[index_range_end >= len(input_id)] = len(input_id)
            index_range_starts.append(index_range_start)
            index_range_ends.append(index_range_end)

        def _split_embeddings(input_ids, niter, iter_len, embeddings,
                              embedding_ranges):
            embs = [None] * niter
            ranges = [None] * niter

            if embeddings is None:
                return embs, ranges

            for i in range(niter):
                iembs = []
                iranges = []
                for emb, (begin, end) in zip(embeddings, embedding_ranges):
                    assert end <= len(input_ids)
                    if begin >= (i + 1) * iter_len or end <= i * iter_len:
                        continue
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb)
                    emb = emb.squeeze()
                    offx = max(iter_len * i - begin, 0)
                    offy = max(end - iter_len * (i + 1), 0)
                    emb = emb[offx:emb.shape[0] - offy]
                    off = max(begin - iter_len * i, 0)
                    rng = [off, off + emb.shape[0]]
                    iembs.append(emb)
                    iranges.append(rng)

                iembs = iembs or None
                iranges = iranges or None
                embs[i] = iembs
                ranges[i] = iranges

            return embs, ranges

        if input_embeddings is not None:
            if not isinstance(input_embeddings[0], list):
                input_embeddings = [input_embeddings]
                input_embedding_ranges = [input_embedding_ranges]
            _input_embeddings = []
            _input_embedding_ranges = []
            for i in range(len(input_ids)):
                embeddings, ranges = _split_embeddings(
                    input_ids[i], n_max_iter, max_input_len,
                    input_embeddings[i], input_embedding_ranges[i])
                _input_embeddings.append(embeddings)
                _input_embedding_ranges.append(ranges)
            input_embeddings = _input_embeddings
            input_embedding_ranges = _input_embedding_ranges

        logits = []
        generator = self.engine.create_instance()
        for i in range(n_max_iter):
            steps = [start[i] for start in index_range_starts]
            _input_ids = [
                input_id[start[i]:end[i]] for input_id, start, end in zip(
                    input_ids, index_range_starts, index_range_ends)
            ]
            embeddings = None
            ranges = None
            if input_embeddings is not None:
                embeddings = [x[i] for x in input_embeddings]
                ranges = [x[i] for x in input_embedding_ranges]

            _logits = generator.decode(_input_ids,
                                       steps=steps,
                                       input_embeddings=embeddings,
                                       input_embedding_ranges=ranges,
                                       sequence_start=(i == 0),
                                       sequence_end=(i == n_max_iter - 1))
            _logits = _logits.cpu()
            logits.append(_logits)

        # concat logits. Shape is [bsz, seq_len, vocab_size]
        logits = torch.cat(logits, dim=1)
        return logits

    def get_ppl(self, input_ids: Union[List[int],
                                       List[List[int]]]) -> List[float]:
        """Get perplexity scores given a list of input tokens that have to be
        of the same length.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids

        Returns:
            Union[float, List[float]]: A list of perplexity scores.
        """
        assert isinstance(input_ids, List)
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        generator = self.engine.create_instance()

        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.hf_tm_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (vocab_size * 4)
        sizes = [len(_) for _ in input_ids]
        losses = []
        target_counts = []
        sorted_index_values = sorted(list(enumerate(sizes)),
                                     key=lambda x: x[1],
                                     reverse=True)
        sizes = [value for index, value in sorted_index_values]
        indices = [index for index, value in sorted_index_values]
        logger.info(f'sorted sizes: {sizes}')
        logger.info(f'sorted indices: {indices}')
        for (start, end) in self._batch_iterator(sizes, max_input_len):
            logger.info(f'start: {start}, end: {end}')
            if start == end:
                _input_ids = input_ids[indices[start]]
                loss, target_count = self._get_long_text_ppl(
                    generator=generator,
                    input_ids=_input_ids,
                    max_input_len=max_input_len)
                losses.append(loss)
                target_counts.append(target_count)
            else:
                _input_ids = [input_ids[indices[i]] for i in range(start, end)]
                loss, target_count = self._get_ppl(
                    generator=generator,
                    input_ids=_input_ids,
                    max_input_len=max_input_len,
                )
                losses.append(loss)
                target_counts.append(target_count)
        loss = torch.concatenate(losses)
        target_count = torch.concatenate(target_counts)
        loss_avg = loss / target_count
        loss_avg = loss_avg.numpy().tolist()
        result = list(range(len(loss_avg)))
        for index, sorted_index in enumerate(indices):
            result[sorted_index] = loss_avg[index]
        return result

    def _batch_iterator(self, sizes, max_value):
        """Return an iterator that calculates intervals (start, end) of a
        descend-order list, in which the sum of values in the range is the
        maximum number not less than max_value. By "the sum of values",

        here it means $$len(sizes[start:end]) * sizes[start]$$
        """
        i = 0
        while i < len(sizes):
            current_sum = 0
            start_index = i

            while i < len(
                    sizes) and current_sum + sizes[start_index] <= max_value:
                current_sum += sizes[start_index]
                i += 1

            yield (start_index, i)
            if i > start_index:
                continue
            else:
                i += 1

    def _get_long_text_ppl(self, generator, input_ids, max_input_len):
        assert all(isinstance(_, int) for _ in input_ids)
        seq_len = len(input_ids)
        assert seq_len > max_input_len
        logger.info(f'get long text ppl: seq_len {seq_len}')

        losses = []
        target_counts = []
        for i in range(0, seq_len, max_input_len):
            token_ids = input_ids[i:i + max_input_len]
            step = [i]
            # shift token_ids by 1 to the left
            target_ids = input_ids[i + 1:i + 1 + max_input_len]

            loss, target_count = self._get_ppl(
                generator=generator,
                input_ids=[token_ids],
                max_input_len=max_input_len,
                target_ids=[target_ids],
                steps=step,
                sequence_start=(i == 0),
                sequence_end=(i + max_input_len >= seq_len))
            losses.append(loss)
            target_counts.append(target_count)
        loss_sum = torch.concatenate(losses).sum().unsqueeze(0)
        target_count = torch.concatenate(target_counts).sum().unsqueeze(0)
        return loss_sum, target_count

    def _get_ppl(self,
                 generator,
                 input_ids,
                 max_input_len,
                 target_ids=None,
                 steps=None,
                 sequence_start: bool = True,
                 sequence_end: bool = True):
        assert isinstance(input_ids, List)
        assert all(isinstance(_, List) for _ in input_ids)
        if target_ids:
            assert all(isinstance(_, List) for _ in target_ids)

        lens = [len(_) for _ in input_ids]
        total_len = sum(lens)
        assert sum(lens) <= max_input_len

        logger.info(f'get_ppl: bs: {len(input_ids)}, lens: {lens}, '
                    f'total_len: {total_len}')
        torch.cuda.empty_cache()
        logits = generator.decode(input_ids=input_ids,
                                  steps=steps,
                                  sequence_start=sequence_start,
                                  sequence_end=sequence_end)
        bsz, seq_len, vocab_size = logits.shape
        logits = logits.float()
        padding_token_id = -100
        if target_ids is None:
            # shift token_ids by 1 to the left
            target_ids = [x[1:] + [padding_token_id] for x in input_ids]
        else:
            target_ids = [
                target_ids[i] + [padding_token_id]
                if len(target_ids[i]) < len(input_ids[i]) else target_ids[i]
                for i in range(bsz)
            ]
        target_ids = [
            torch.Tensor(torch.LongTensor(_target_ids))
            for _target_ids in target_ids
        ]
        target_ids = pad_sequence(target_ids,
                                  batch_first=True,
                                  padding_value=padding_token_id)
        target_ids = target_ids.to(logits.device)
        target_mask = target_ids != padding_token_id

        # compute cross entropy loss
        flat_logits = logits.contiguous().view(-1, vocab_size)
        flat_target_ids = target_ids.contiguous().view(-1)
        flat_loss_matrix = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_target_ids,
            reduction='none',
            ignore_index=padding_token_id)
        flat_loss_matrix = flat_loss_matrix.view(bsz, seq_len)
        loss = flat_loss_matrix.sum(dim=-1).cpu()
        target_count = target_mask.sum(dim=-1).cpu()
        return loss, target_count



YIJIAN_SYSTEM_PROMPT = "我是来自百度的多模态大模型一见大模型，英文名是Yijian。"

def filter_text(text: str) -> str:
    """
    Filter out sensitive information in the text.
    """
    # Special case handling for simple inputs
    logger.info(f"Original response content: {text}")
    if text.strip() in ['0'] or text.startswith('{'):
        return text

    filter_keywords = ['上海人工智能实验室', 'OpenGVLab', '商汤科技', 'Shanghai AI Lab', 'SenseTime']

    import re
    # Check if text contains any filter keywords and no second sentence
    if any(keyword in text for keyword in filter_keywords) and not re.search(r'2\..*', text):
        return YIJIAN_SYSTEM_PROMPT

    sentences = re.split('([,，。！？\n])', text)
    filtered_sentences = []

    i = 0
    sentence_count = 1
    while i < len(sentences):
        sentence = sentences[i]

        if sentence:
            should_filter = any(keyword in sentence for keyword in filter_keywords)

            if not should_filter:
                processed_sentence = sentence
                processed_sentence = re.sub(
                    r'internvl|InternVL|书生多模态大模型',
                    'Yijian',
                    processed_sentence,
                    flags=re.IGNORECASE
                )

                processed_sentence = re.sub(
                    r'文心一言',
                    '一见多模态大模型',
                    processed_sentence
                )

                # Clean up any existing numbers at start
                cleaned_sentence = re.sub(r'^\d+[\s.]*', '', processed_sentence)

                # For first non-empty sentence
                if not filtered_sentences:
                    if '百度' in cleaned_sentence:
                        filtered_sentences.append(f"1.{cleaned_sentence}")
                    elif not re.search(r'我是(Yijian|一见)', cleaned_sentence):
                        filtered_sentences.append(f"1.我是Yijian，{cleaned_sentence}")
                    else:
                        filtered_sentences.append(f"1.{cleaned_sentence}")
                else:
                    # Add number prefix for new main sentences
                    if i > 0 and sentences[i - 1] in ['。', '！', '？']:
                        filtered_sentences.append(f"{sentence_count + 1}.{cleaned_sentence}")
                        sentence_count += 1
                    else:
                        filtered_sentences.append(cleaned_sentence)

                # Handle punctuation
                if i + 1 < len(sentences):
                    next_punct = sentences[i + 1]
                    if filtered_sentences and re.search(r'我是(Yijian|一见)$', filtered_sentences[-1]):
                        filtered_sentences.append('。')
                    else:
                        filtered_sentences.append(next_punct)

        i += 2

    if not filtered_sentences:
        return YIJIAN_SYSTEM_PROMPT

    # If only one sentence ends with "我是Yijian" or similar, add period
    if len(filtered_sentences) == 2 and re.search(r'我是(Yijian|一见)$', filtered_sentences[0]):
        filtered_sentences[1] = "。"

    result = ''.join(filtered_sentences)

    # Ensure proper ending
    if result[-1] in ["，", ","]:
        result = result[:-1] + "。"

    # Clean up punctuation
    result = re.sub(r'[,，。]{2,}', '。', result)
    result = re.sub(r'。\s*', '。', result)

    # Replace comma after "我是Yijian" or "我是一见" with period
    result = re.sub(r'(我是(?:Yijian|一见))[,，]', r'\1。', result)

    return result

if __name__ == '__main__':
    # Test cases
    text1 = "1.我是InternVL,是由上海人工智能实验室的通用视觉团队(OpenGVLab)和商汤科技联合开发的模型。2.我的开发语言和技术框架是基于深度学习技术，使用了Transformer架构和大规模预训练模型。"
    print("---------------1----------------")
    print(filter_text(text1))

    text2 = "1.我是InternVL，是由上海人工智能实验室的通用视觉团队(OpenGVLab)和商汤科技联合开发的模型。"
    print("---------------2----------------")
    print(filter_text(text2))

    text3 = "1.我是InternVL,是由上海人工智能实验室的通用视觉团队(OpenGVLab)和商汤科技联合开发的模型。"
    print("---------------3----------------")
    print(filter_text(text3))

    text4 = "1.我是一见,是由上海人工智能实验室的通用视觉团队(OpenGVLab)和商汤科技联合开发的模型。2.我的开发语言和技术框架是基于深度学习技术，使用了Transformer架构和大规模预训练模型。"
    print("---------------4----------------")
    print(filter_text(text4))

    text5 = "1.我是是由上海人工智能实验室的通用视觉团队(OpenGVLab)和商汤科技联合开发的模型。2.我的开发语言和技术框架是基于深度学习技术，使用了Transformer架构和大规模预训练模型。"
    print("---------------5----------------")
    print(filter_text(text5))

    text6 = "1.我是由上海人工智能实验室的通用视觉团队(OpenGVLab)和商汤科技联合开发的模型。2.我的开发语言和技术框架是基于深度学习技术，使用了Transformer架构和大规模预训练模型。"
    print("---------------6----------------")
    print(filter_text(text6))

    text7 = "1.我是由上海人工智能实验室的通用视觉团队(OpenGVLab)和商汤科技"
    print("---------------7----------------")
    print(filter_text(text7))

    text8 = "0"
    print("---------------8----------------")
    print(filter_text(text8))

    text9 = "{\"图中是否有有人\":\"Yes\"}"
    print("---------------9----------------")
    print(filter_text(text9))

    text10= "1. 我是百度文心一言，是由百度公司开发的一款基于深度学习技术的自然语言处理模型。2. 我的技术框架是基于百度自主研发的深度学习平台飞桨（PaddlePaddle）。飞桨提供了丰富的深度学习算法和工具，支持多种编程语言，包括Python、C++等。"
    print("---------------10----------------")
    print(filter_text(text10))