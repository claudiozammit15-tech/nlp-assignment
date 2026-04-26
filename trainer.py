from typing import Dict, List, Union

import torch
from trl import DPOTrainer
from trl.trainer.utils import pad_to_length

class TDPOTrainerPadtoMul8(DPOTrainer):
    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]], padding_value: int
    ) -> Dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
        and completion sequences.

        Args:
            batch (`Dict[str, Union[List, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `Dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen
            or rejected sequences.
        """
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
            )

        # Concatenate the chosen and rejected completions
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        if (output["prompt_input_ids"].shape[1]+max_completion_length)%8 != 0:
            max_completion_length=((output["prompt_input_ids"].shape[1]+max_completion_length)//8+1)*8-output["prompt_input_ids"].shape[1]
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return output
