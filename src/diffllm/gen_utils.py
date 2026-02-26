from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _find_pattern_matches(
    seq: List[int],
    patterns: List[List[int]],
    start: int,
    end: int,
) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    if start >= end:
        return matches
    for pat in patterns:
        if not pat:
            continue
        pat_len = len(pat)
        limit = end - pat_len + 1
        for i in range(start, limit):
            if seq[i : i + pat_len] == pat:
                matches.append((i, pat_len))
    matches.sort()
    return matches


def parse_structure_blocks(
    seq: List[int],
    start: int,
    end: int,
    structure_cfg: Dict,
) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    think_start = structure_cfg.get("think_start_tags", [])
    think_end = structure_cfg.get("think_end_tags", [])
    summary_start = structure_cfg.get("summary_start_tag", [])
    summary_end = structure_cfg.get("summary_end_tag", [])

    think_starts = _find_pattern_matches(seq, think_start, start, end)
    think_ends = _find_pattern_matches(seq, think_end, start, end)
    summary_starts = _find_pattern_matches(seq, summary_start, start, end)
    summary_ends = _find_pattern_matches(seq, summary_end, start, end)

    struct_spans: List[Tuple[int, int]] = []
    think_blocks: List[Tuple[int, int]] = []

    end_idx = 0
    for s_pos, s_len in think_starts:
        while end_idx < len(think_ends) and think_ends[end_idx][0] <= s_pos:
            end_idx += 1
        if end_idx >= len(think_ends):
            break
        e_pos, e_len = think_ends[end_idx]
        content_start = s_pos + s_len
        content_end = e_pos
        if content_start < content_end:
            think_blocks.append((content_start, content_end))
        struct_spans.append((s_pos, s_pos + s_len))
        struct_spans.append((e_pos, e_pos + e_len))
        end_idx += 1

    summary_block = None
    if summary_starts and summary_ends:
        s_pos, s_len = summary_starts[0]
        e_pos, e_len = None, None
        for pos, length in summary_ends:
            if pos > s_pos:
                e_pos, e_len = pos, length
                break
        if e_pos is not None:
            content_start = s_pos + s_len
            content_end = e_pos
            if content_start < content_end:
                summary_block = (content_start, content_end)
            struct_spans.append((s_pos, s_pos + s_len))
            struct_spans.append((e_pos, e_pos + e_len))

    return think_blocks, summary_block, struct_spans


def shuffle_think_blocks(
    input_ids: torch.Tensor,
    maskable_mask: torch.Tensor,
    structure_cfg: Optional[Dict],
) -> torch.Tensor:
    if not structure_cfg or not structure_cfg.get("shuffle_think", False):
        return input_ids

    think_start = structure_cfg.get("think_start_tags", [])
    summary_start = structure_cfg.get("summary_start_tag", [])
    if not think_start:
        return input_ids

    output = input_ids.clone()
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        mask_idx = maskable_mask[i].nonzero(as_tuple=True)[0]
        if mask_idx.numel() == 0:
            continue
        resp_start = mask_idx[0].item()
        resp_end = mask_idx[-1].item() + 1

        seq = input_ids[i].tolist()
        think_starts = _find_pattern_matches(seq, think_start, resp_start, resp_end)
        if len(think_starts) < 2:
            continue
        summary_starts = _find_pattern_matches(seq, summary_start, resp_start, resp_end)
        shuffle_end = summary_starts[0][0] if summary_starts else resp_end
        think_positions = [pos for pos, _ in think_starts if pos < shuffle_end]
        if len(think_positions) < 2:
            continue

        block_spans = []
        for idx, pos in enumerate(think_positions):
            next_pos = (
                think_positions[idx + 1] if idx + 1 < len(think_positions) else shuffle_end
            )
            if pos < next_pos:
                block_spans.append((pos, next_pos))

        if len(block_spans) < 2:
            continue

        prefix = seq[resp_start:block_spans[0][0]]
        suffix = seq[shuffle_end:resp_end]
        blocks = [seq[s:e] for s, e in block_spans]
        perm = torch.randperm(len(blocks)).tolist()
        shuffled = []
        for idx in perm:
            shuffled.extend(blocks[idx])
        new_resp = prefix + shuffled + suffix
        if len(new_resp) != resp_end - resp_start:
            continue
        output[i, resp_start:resp_end] = torch.tensor(
            new_resp, device=input_ids.device, dtype=input_ids.dtype
        )
    return output


def q_sample(
    input_ids,
    maskable_mask,
    mask_token_id,
    min=0.0,
    max=1.0,
    eos_token_id=None,
    t=None,
    t_mask=None,
    structure_cfg: Optional[Dict] = None,
):
    x_0 = input_ids

    if t_mask is None:
        if t is None:
            t = torch.rand((x_0.shape[0],), dtype=torch.float, device=input_ids.device)
            t = min + (max - min) * t
        if not structure_cfg:
            u = torch.rand_like(x_0, dtype=torch.float)  # t/T prob to mask
            t_mask = (u < t[:, None]) & maskable_mask
        else:
            t_mask = torch.zeros_like(x_0, dtype=torch.bool)
            batch_size, seq_len = x_0.shape
            device = x_0.device

            r_max = float(structure_cfg.get("rpim_r_max", 0.9))
            alpha = float(structure_cfg.get("rpim_alpha", 0.8))
            kappa_think = float(structure_cfg.get("rpim_kappa_think", 0.0))
            kappa_summary = float(structure_cfg.get("rpim_kappa_summary", 1.0))

            for i in range(batch_size):
                mask_idx = maskable_mask[i].nonzero(as_tuple=True)[0]
                if mask_idx.numel() == 0:
                    continue
                resp_start = mask_idx[0].item()
                resp_end = mask_idx[-1].item() + 1

                seq = x_0[i].tolist()
                think_blocks, summary_block, struct_spans = parse_structure_blocks(
                    seq, resp_start, resp_end, structure_cfg
                )

                struct_mask = torch.zeros((seq_len,), device=device, dtype=torch.bool)
                for s, e in struct_spans:
                    struct_mask[s:e] = True

                r = float(t[i].item()) * r_max
                if r > 1.0:
                    r = 1.0
                p_mask = torch.zeros((seq_len,), device=device, dtype=torch.float)

                for b_start, b_end in think_blocks:
                    length = b_end - b_start
                    if length <= 0:
                        continue
                    if length == 1 or kappa_think == 0:
                        p_mask[b_start:b_end] = torch.maximum(
                            p_mask[b_start:b_end],
                            torch.full((length,), r, device=device),
                        )
                    else:
                        z = torch.linspace(0, 1, steps=length, device=device)
                        p = torch.clamp(r * (1 + kappa_think * z), 0.0, 1.0)
                        p_mask[b_start:b_end] = torch.maximum(p_mask[b_start:b_end], p)

                if summary_block is not None:
                    b_start, b_end = summary_block
                    length = b_end - b_start
                    if length > 0:
                        g = 1.0 - float(t[i].item())
                        if g < alpha:
                            p_mask[b_start:b_end] = 1.0
                        elif length == 1 or kappa_summary == 0:
                            p_mask[b_start:b_end] = torch.maximum(
                                p_mask[b_start:b_end],
                                torch.full((length,), r, device=device),
                            )
                        else:
                            z = torch.linspace(0, 1, steps=length, device=device)
                            p = torch.clamp(r * (1 + kappa_summary * z), 0.0, 1.0)
                            p_mask[b_start:b_end] = torch.maximum(
                                p_mask[b_start:b_end], p
                            )

                fallback_mask = (p_mask == 0) & maskable_mask[i] & ~struct_mask
                if fallback_mask.any():
                    p_mask[fallback_mask] = r

                u = torch.rand((seq_len,), device=device, dtype=torch.float)
                t_mask[i] = (u < p_mask) & maskable_mask[i] & ~struct_mask

    x_t = x_0.masked_fill(t_mask, mask_token_id)

    if eos_token_id is not None:
        # get the last non-eos token index
        last_non_eos_token_idx = ((input_ids != eos_token_id) | (~maskable_mask)).sum(
            dim=-1
        ) - 1
        seq_len = x_0.shape[1]

        for i in range(x_0.shape[0]):
            if last_non_eos_token_idx[i] < seq_len - 1:  # with eos tokens
                t_mask_at_eos = t_mask[
                    i, last_non_eos_token_idx[i] + 1
                ]  # use arbitrary eos token
                # t_mask[i, last_non_eos_token_idx[i] + 2:] = False  # only learn the first eos token
                if t_mask_at_eos:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = mask_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = True
                else:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = eos_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = False

    return x_t, t, t_mask  #  True means it's "MASK" token and should have loss


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0
