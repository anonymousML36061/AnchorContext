import torch

# ensures that all elements of value are included in local_value, even when dimension_size is not divisible by world_size
# def extract_local(value, rank, world_size, device, dim=1):
#     dimension_size = value.shape[dim]
#     base_length = dimension_size // world_size
#     extra = dimension_size % world_size

#     # Calculate the start and end indices for each rank
#     if rank < extra:
#         sub_seq_length = base_length + 1
#         sub_seq_start = rank * sub_seq_length
#     else:
#         sub_seq_length = base_length
#         sub_seq_start = rank * sub_seq_length + extra

#     sub_seq_end = sub_seq_start + sub_seq_length

#     # Create a slice object for the desired dimension
#     slices = [slice(None)] * value.dim()
#     slices[dim] = slice(sub_seq_start, sub_seq_end)

#     local_value = value[tuple(slices)]

#     return local_value.to(device)


def extract_local(value, rank, world_size, device, dim=1):
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]

    return local_value.to(device)


def extract_global(value, rank, world_size, device, dim=1):
    return value.to(device)


def prepare_ulysses_attn_inputs(
    input_ids, position_ids, target_ids, doc_ids, global_position_ids, rank, world_size, device
):
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )

    global_doc_ids = extract_global(
        doc_ids,
        rank,
        world_size,
        device,
    )

    global_position_ids = extract_global(
        global_position_ids,
        rank,
        world_size,
        device,
    )

    if target_ids is not None:
        local_target_ids = extract_local(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
        "global_doc_ids": global_doc_ids,
        "global_position_ids": global_position_ids,
    }
